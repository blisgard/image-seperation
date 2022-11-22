import exiftool


def write_custom_tags(tag_name, path, description):

    etPath = "exiftool"
    cfgFile = "./exif.config"
    et = exiftool.ExifTool(etPath, config_file=cfgFile)

    with et:
        try:
            encoded_tag = "".join(" -EXIF:" + tag_name + "=" + description).encode()
            encoded_path = "".join(path).encode()
            et.execute(encoded_tag, encoded_path)
        except Exception as e:
            print(e)



