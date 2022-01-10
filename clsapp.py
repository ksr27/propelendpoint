import requests

import streamlit as st

import matplotlib.pyplot as plt

from PIL import Image

from io import BytesIO

from torchvision import models

from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image

from torchcam import methods

from torchcam.methods._utils import locate_candidate_layer

from torchcam.utils import overlay_mask

import torch

CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM", "LayerCAM"]

TV_MODELS = ["densenet169"]

LABEL_MAP = requests.get(

    "https://raw.githubusercontent.com/ksr27/streamlit-example/master/labels.json"

).json()

def main():

    # Wide mode

    st.set_page_config(layout="wide")

    # Designing the interface

    st.title("Propelwise Analytics: Rad_fw -> Classification")

    # For newline

    st.write('\n')

    st.write('Check the project at: https://github.com/propelwise/Rad_fw')

    # For newline

    st.write('\n')

    # Set the columns

    cols = st.columns((1, 1, 1))

    cols[0].header("Input image")

    cols[1].header("CAM")

    cols[-1].header("Overlayed CAM")

    # Sidebar

    # File selection

    st.sidebar.title("Input selection")

    # Disabling warning

    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Choose your own image

    uploaded_file = st.sidebar.file_uploader("Upload files", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:

        img = Image.open(BytesIO(uploaded_file.read()), mode='r').convert('RGB')

        cols[0].image(img, use_column_width=True)

    # Model selection

    st.sidebar.title("Setup")

    tv_model = st.sidebar.selectbox("Classification model", TV_MODELS)

    default_layer = ""

    if tv_model is not None:

        with st.spinner('Loading model...'):

            model = models.__dict__[tv_model](pretrained=True)#.eval()
            model.load_state_dict(torch.load('Dense169.pt',map_location=torch.device('cpu')))
            # model.load_state_dict(torch.load('/content/drive/MyDrive/Buddi/CT CC/Dense169.pt',map_location=torch.device('cpu')))
            model.eval()
        default_layer = locate_candidate_layer(model, (3, 224, 224))

    target_layer = st.sidebar.text_input("Target layer", default_layer)

    cam_method = st.sidebar.selectbox("CAM method", CAM_METHODS)

    if cam_method is not None:

        cam_extractor = methods.__dict__[cam_method](

            model,

            target_layer=target_layer.split("+") if len(target_layer) > 0 else None

        )

    class_choices = [f"{idx + 1} - {class_name}" for idx, class_name in enumerate(LABEL_MAP)]

    class_selection = st.sidebar.selectbox("Class selection", ["Predicted class (argmax)"] + class_choices)

    # For newline

    st.sidebar.write('\n')

    if st.sidebar.button("Compute CAM"):

        if uploaded_file is None:

            st.sidebar.error("Please upload an image first")

        else:

            with st.spinner('Analyzing...'):

                # Preprocess image

                img_tensor = normalize(to_tensor(resize(img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                # Forward the image to the model

                out = model(img_tensor.unsqueeze(0))

                # Select the target class

                if class_selection == "Predicted class (argmax)":

                    class_idx = out.squeeze(0).argmax().item()

                else:

                    class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])

                # Retrieve the CAM

                cams = cam_extractor(class_idx, out)

                # Fuse the CAMs if there are several

                cam = cams[0] if len(cams) == 1 else cam_extractor.fuse_cams(cams)

                # Plot the raw heatmap

                fig, ax = plt.subplots()

                ax.imshow(cam.numpy())

                ax.axis('off')

                cols[1].pyplot(fig)

                # Overlayed CAM

                fig, ax = plt.subplots()

                result = overlay_mask(img, to_pil_image(cam, mode='F'), alpha=0.5)

                ax.imshow(result)

                ax.axis('off')

                cols[-1].pyplot(fig)

if __name__ == '__main__':

    main()

