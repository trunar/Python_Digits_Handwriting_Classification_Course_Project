from PIL import Image
import os


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        output_folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("RGBA")

                new_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
                new_img.paste(img, (0, 0), img)

                new_img.save(os.path.join(output_folder_path, filename))


input_directory = 'transparentdataset'
output_directory = 'dataset'

process_images(input_directory, output_directory)
