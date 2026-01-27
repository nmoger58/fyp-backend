import os
import torch
import numpy as np

def preprocess_single_video(
    video_path,
    video_name,
    face_extractor,
    num_frames=16,
    save_dir="face_cache",
    device=None
):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU info
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{video_name}.npy")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"Already processed: {output_path}")
        return output_path

    try:
        faces = face_extractor.extract_faces_from_video(
            video_path,
            num_frames=num_frames,
            device=device
        )

        if faces is None:
            print("❌ No faces detected")
            return None

        np.save(output_path, faces)
        print(f"✅ Saved preprocessed faces to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
