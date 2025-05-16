import cv2
import os
import numpy as np


def images_to_video(image_folder, output_video_path, fps):
    """
    Convert a sequence of images from a folder into a video file.

    Parameters:
    image_folder (str): Path to the folder containing the images.
    output_video_path (str): Path to the output video file.
    fps (int): Frames per second for the output video.
    """
    # List all image files in the folder with appropriate extensions
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure images are sorted to maintain the correct order in the video
    images.sort()

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Resize the frame if it has different dimensions than the first image
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # Write the frame to the video file
        video.write(frame)

    # Release the video writer and close any open windows
    cv2.destroyAllWindows()
    video.release()

nx = 128
comp_rad = 0.2
if __name__ == "__main__":
    # Define base folder and velocity field name
    base_folder = f"Logistic/Codes/simulation_results/{nx}_R{comp_rad:.1f}"  # Replace with the actual base folder path
    velocity_field_name = 'rankine'  # Replace with the actual velocity field name

    # Define parameters for the video creation
    mu = 400
    Pe = 25
    w = 0# 2 * np.pi

    # Construct the paths for the image folder and the output video file
    image_folder_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{Pe:.1f}_w{w:.2f}'
    output_video_path = f'{base_folder}/{velocity_field_name}_video{mu:.0f}_{Pe:.0f}.mov'

    # Set the desired frames per second (fps) for the output video
    fps = 30

    # Call the function to convert images to video
    images_to_video(image_folder_path, output_video_path, fps)
