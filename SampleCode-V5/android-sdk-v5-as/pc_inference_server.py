#!/usr/bin/env python3
"""
PC Inference Server for DJI Drone AutoPilot
Receives real-time camera frames (JPEG) from Android app over HTTP,
runs PyTorch MobileNet inference on PC (GPU/CPU),
and returns predicted VirtualStick control values [lv, lh, rv, rh] in JSON.
"""

import argparse
import io
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MapMatchingPolicy(nn.Module):
    """Dual image (Current + Reference) + stick history -> 4 stick channels"""
    def __init__(self, history_len: int = 5, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        history_dim = history_len * 4
        self.head = nn.Sequential(
            nn.Linear(1280 * 2 + history_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def forward(self, x, ref_x, history):
        feat = self.flatten(self.pool(self.features(x)))
        ref_feat = self.flatten(self.pool(self.features(ref_x)))
        combined = torch.cat([feat, ref_feat, history], dim=1)
        return self.head(combined)


class BaselinePolicy(nn.Module):
    """Pure E2E Baseline Policy: Single camera image + stick history -> 4 stick channels"""
    def __init__(self, history_len: int = 5, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        history_dim = history_len * 4
        self.head = nn.Sequential(
            nn.Linear(1280 + history_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Tanh(),
        )

    def forward(self, x, history):
        feat = self.flatten(self.pool(self.features(x)))
        combined = torch.cat([feat, history], dim=1)
        return self.head(combined)


# Global Server State
MODEL = None
DEVICE = "cpu"
MODEL_TYPE = "baseline"
HISTORY_LEN = 5
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_pytorch_model(checkpoint_path, model_type="auto", device="cpu"):
    global MODEL, MODEL_TYPE, HISTORY_LEN, DEVICE

    DEVICE = device
    print(f"Loading checkpoint '{checkpoint_path}' on device '{device}'...")

    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running in mock dummy model mode.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    HISTORY_LEN = checkpoint.get("history_len", 5)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if model_type == "auto":
        is_baseline = "baseline" in checkpoint_path.lower() or ("head.0.weight" in state_dict and state_dict["head.0.weight"].shape[1] < 2000)
        MODEL_TYPE = "baseline" if is_baseline else "map_matching"
    else:
        MODEL_TYPE = model_type

    if MODEL_TYPE == "baseline":
        model = BaselinePolicy(history_len=HISTORY_LEN, freeze_backbone=True)
    else:
        model = MapMatchingPolicy(history_len=HISTORY_LEN, freeze_backbone=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    MODEL = model
    print(f"Successfully loaded PyTorch MobileNet ({MODEL_TYPE} mode) on {device}!")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for low-latency streaming"""
    daemon_threads = True


class InferenceHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({"status": "ok", "model_loaded": MODEL is not None, "device": DEVICE, "model_type": MODEL_TYPE})
            self.wfile.write(resp.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            start_time = time.time()
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length)

            # 1. Parse Stick History Header
            hist_str = self.headers.get("X-Stick-History", "")
            if hist_str:
                hist_vals = [float(v) for v in hist_str.split(",")]
            else:
                hist_vals = [0.0] * (HISTORY_LEN * 4)

            # Ensure 20 floats
            if len(hist_vals) < HISTORY_LEN * 4:
                hist_vals += [0.0] * (HISTORY_LEN * 4 - len(hist_vals))

            # 2. Decode JPEG Image
            has_ref = self.headers.get("X-Has-Ref-Image", "0") == "1"
            curr_len = int(self.headers.get("X-Curr-Img-Len", content_length))

            curr_bytes = request_body[:curr_len]
            ref_bytes = request_body[curr_len:] if has_ref else None

            try:
                curr_img = Image.open(io.BytesIO(curr_bytes)).convert("RGB")
                curr_tensor = TRANSFORM(curr_img).unsqueeze(0).to(DEVICE)
            except Exception as e:
                print(f"[Error] Failed to decode image: {e}")
                self.send_error(400, f"Invalid JPEG image: {e}")
                return

            hist_tensor = torch.tensor([hist_vals], dtype=torch.float32).to(DEVICE)

            # 3. Model Inference
            if MODEL is not None:
                with torch.no_grad():
                    if MODEL_TYPE == "map_matching" and ref_bytes and len(ref_bytes) > 0:
                        ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
                        ref_tensor = TRANSFORM(ref_img).unsqueeze(0).to(DEVICE)
                        output = MODEL(curr_tensor, ref_tensor, hist_tensor)
                    elif MODEL_TYPE == "map_matching":
                        # Fallback dummy ref image if missing
                        dummy_ref = torch.zeros_like(curr_tensor)
                        output = MODEL(curr_tensor, dummy_ref, hist_tensor)
                    else:
                        output = MODEL(curr_tensor, hist_tensor)

                    out_data = output.cpu().squeeze(0).tolist()
            else:
                # Mock output if model file not found
                out_data = [0.0, 0.0, 0.53, 0.0]

            # Scale output [-1, 1] to [-660, 660]
            sticks = [
                int(max(-660, min(660, out_data[0] * 660))),
                int(max(-660, min(660, out_data[1] * 660))),
                int(max(-660, min(660, out_data[2] * 660))),
                int(max(-660, min(660, out_data[3] * 660))),
            ]

            infer_ms = (time.time() - start_time) * 1000.0

            print(f"[PC Predict] Infer: {infer_ms:.1f}ms | Raw: {[round(v,3) for v in out_data]} -> Sticks: LV={sticks[0]}, LH={sticks[1]}, RV={sticks[2]}, RH={sticks[3]}")

            # 4. Respond with JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response_data = {
                "status": "ok",
                "sticks": sticks,
                "inference_time_ms": round(infer_ms, 1)
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard HTTP GET/POST log spam
        return


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to external address (packets aren't sent) to find LAN routing interface IP
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    parser = argparse.ArgumentParser(description="PC Inference HTTP Server for DJI Drone Autonomous Flight")
    parser.add_argument("--model", default="baseline_model.pt", help="PyTorch model checkpoint (.pt)")
    parser.add_argument("--type", choices=["auto", "baseline", "map_matching"], default="auto", help="Model type")
    parser.add_argument("--port", type=int, default=5000, help="HTTP Server Port (default: 5000)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    load_pytorch_model(args.model, args.type, args.device)

    server_address = ("0.0.0.0", args.port)
    httpd = ThreadedHTTPServer(server_address, InferenceHTTPRequestHandler)

    local_ip = get_local_ip()
    print("==========================================================")
    print(f"🚀 PC Inference Server RUNNING on http://0.0.0.0:{args.port}")
    print(f"   Local IP Address for Tablet: http://{local_ip}:{args.port}")
    print(f"   Model: {args.model} ({MODEL_TYPE}) on {args.device.upper()}")
    print("==========================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down PC Inference Server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
