import email
import os
import sys


def extract_images_from_eml(eml_path, output_dir="attachments"):
    if not os.path.isfile(eml_path):
        print(f"Erreur : fichier '{eml_path}' introuvable.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(eml_path, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)

    found = False

    for part in msg.walk():
        content_type = part.get_content_type()

        if content_type.startswith("image/"):
            filename = part.get_filename()

            if filename is None:
                ext = content_type.split("/")[1]
                filename = f"image_extraite.{ext}"

            image_data = part.get_payload(decode=True)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "wb") as img_file:
                img_file.write(image_data)

            print(f"Image extraite : {output_path}")
            found = True
            break

    if not found:
        print("Aucune image trouvée dans les pièces jointes du mail.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python extract_attachment.py <fichier.eml>")
        sys.exit(1)

    eml_file = sys.argv[1]
    extract_images_from_eml(eml_file)