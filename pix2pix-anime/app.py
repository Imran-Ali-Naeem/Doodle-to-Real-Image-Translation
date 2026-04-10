import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# ── Generator Definition ─────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = EncoderBlock(3,   64,  normalize=False)
        self.e2 = EncoderBlock(64,  128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)
        self.e8 = EncoderBlock(512, 512, normalize=False)

        self.d1 = DecoderBlock(512,  512, dropout=True)
        self.d2 = DecoderBlock(1024, 512, dropout=True)
        self.d3 = DecoderBlock(1024, 512, dropout=True)
        self.d4 = DecoderBlock(1024, 512)
        self.d5 = DecoderBlock(1024, 256)
        self.d6 = DecoderBlock(512,  128)
        self.d7 = DecoderBlock(256,  64)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)
        return self.out(d7)


# ── Load Model ───────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetGenerator().to(DEVICE)
state_dict = torch.load("G_epoch45.pth", map_location=DEVICE)

# handle DataParallel keys
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# ── Inference Function ───────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def colorize(sketch_img):
    tensor = transform(sketch_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
    result = (output[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5).clip(0,1)
    return Image.fromarray((result * 255).astype("uint8"))

# ── Gradio App ───────────────────────────────
app = gr.Interface(
    fn=colorize,
    inputs=gr.Image(type="pil", label="Input Sketch"),
    outputs=gr.Image(type="pil", label="Colorized Output"),
    title="Pix2Pix Anime Sketch Colorization",
    description="Upload an anime sketch and the model will colorize it.",
)

app.launch()
