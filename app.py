import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# LOAD CORESIX MODEL
# ==========================================

model = models.resnet50(weights=None)

model.fc = nn.Linear(2048, 1)

model.load_state_dict(
    torch.load(
        "resnet50_dr_baseline.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# GRAD-CAM
# ==========================================

def generate_gradcam(image, explain_positive=True):

    model.eval()

    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    image_tensor = transform(image).unsqueeze(0).to(device)

    model.zero_grad()

    output = model(image_tensor)

    probability = torch.sigmoid(output).item()

    # Explain the predicted class
    if explain_positive:
        target = output[0, 0]
    else:
        target = -output[0, 0]

    target.backward()

    activation = activations[0]
    gradient = activation.grad

    weights = gradient.mean(dim=(2, 3), keepdim=True)

    cam = (weights * activation).sum(dim=1).squeeze()

    cam = torch.relu(cam)

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    cam = cam.detach().cpu().numpy()

    handle.remove()

    # Resize heatmap
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size)

    cam_array = np.array(cam_image) / 255.0

    # Create overlay
    image_array = np.array(image).astype(float) / 255.0

    plt.figure(figsize=(7, 7))
    plt.imshow(image_array)
    plt.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )
    plt.axis("off")

    plt.tight_layout(pad=0)

    # Save figure temporarily
    plt.savefig(
        "/tmp/gradcam.png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return "/tmp/gradcam.png", probability


# ==========================================
# CORESIX PREDICTION
# ==========================================

def coresix_predict(image):

    if image is None:
        return None, "Please upload a retinal fundus image.", None

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    probability = torch.sigmoid(output).item()

    referable = probability >= 0.5

    # Grad-CAM
    gradcam_image, _ = generate_gradcam(
        image,
        explain_positive=referable
    )

    if referable:

        result = f"""
# 🔴 Referable DR Detected

### Predicted probability: **{probability * 100:.2f}%**

The model classified this retinal image as **potentially referable diabetic retinopathy**.

Further professional ophthalmic evaluation is recommended.
"""

    else:

        result = f"""
# 🟢 Non-Referable DR

### Predicted probability: **{probability * 100:.2f}%**

The model classified this retinal image as **non-referable** according to the current screening threshold.

Further clinical evaluation may still be appropriate based on the patient's situation.
"""

    return image, result, gradcam_image


# ==========================================
# CORESIX UI
# ==========================================

css = """

body {
    background: #f6f9fc;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

#hero {
    text-align: center;
    padding: 30px;
}

#hero h1 {
    font-size: 44px;
}

#hero p {
    font-size: 18px;
    color: #5b6b7a;
}

"""

with gr.Blocks(
    title="CoreSix | Explainable Retinal Screening",
    css=css
) as demo:

    gr.HTML("""
    <div id="hero">

        <h1>👁️ CoreSix</h1>

        <p>
        <b>Explainable AI for Retinal Screening</b>
        </p>

        <p>
        AI-assisted diabetic retinopathy screening
        with visual explanation.
        </p>

    </div>
    """)

    gr.Markdown("## 🔬 AI-Assisted Retinal Screening")

    gr.Markdown(
        "Upload a retinal fundus image to begin screening."
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Fundus Image"
            )

            analyze_button = gr.Button(
                "🔍 Analyze with CoreSix",
                variant="primary",
                size="lg"
            )

        with gr.Column():

            original_output = gr.Image(
                label="Analyzed Image"
            )

            result_output = gr.Markdown(
                "Upload an image to begin."
            )

    gr.Markdown("---")

    gr.Markdown("""
    ## 🧠 Why did CoreSix make this prediction?

    Grad-CAM highlights regions of the retinal image
    that influenced the model's prediction.
    """)

    gradcam_output = gr.Image(
        label="Grad-CAM Explanation"
    )

    gr.Markdown("---")

    gr.Markdown("""
    ## ⚙️ How CoreSix Works

    **Fundus Image**
    ↓

    **Image Preprocessing**
    ↓

    **AI Screening Model**
    ↓

    **Risk Prediction**
    ↓

    **Grad-CAM Explanation**
    ↓

    **Referral Support**
    """)

    gr.Markdown("---")

    gr.Markdown("""
    ## 🌱 Why CoreSix?

    🩺 **AI-Assisted Screening**  
    Helps identify potentially referable cases.

    🔍 **Explainable AI**  
    Grad-CAM provides visual insight into the model's decision.

    🌾 **Rural-Focused**  
    Designed with low-resource screening environments in mind.

    ---

    ⚠️ **Important:** CoreSix provides an AI-assisted
    screening result, not a clinical diagnosis.
    Professional ophthalmic evaluation is recommended.
    """)

    analyze_button.click(
        fn=coresix_predict,
        inputs=image_input,
        outputs=[
            original_output,
            result_output,
            gradcam_output
        ]
    )


if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )