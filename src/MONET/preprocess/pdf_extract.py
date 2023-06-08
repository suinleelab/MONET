import argparse
import glob
import json
import os
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import fitz
from tqdm import tqdm


# https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract.py
def extract_fitz_pixmap(doc, xref, smask):
    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except BaseException:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
            "xref": xref,
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    elif "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
            "xref": xref,
        }
    else:
        return {"xref": xref, **doc.extract_image(xref)}


def get_page_image_info(page):
    page_image_info = [
        OrderedDict(
            zip(
                (
                    "xref",
                    "smask",
                    "width",
                    "height",
                    "bpc",
                    "cs-name",
                    "alt. colorspace",
                    "name",
                    "filter",
                    "referencer",
                ),
                image,
            )
        )
        for image in page.get_images(full=True)
    ]
    page_image_info = OrderedDict(
        {i["xref"]: i for i in page_image_info}
    )  # page.get_image_info(xrefs=True)
    page_image_info_ = OrderedDict(
        {i["xref"]: i for i in page.get_image_info(xrefs=True)}
    )  # page.get_image_info(xrefs=True)
    for xref, image_info in page_image_info_.items():
        if xref in page_image_info:
            page_image_info[xref].update(image_info)
            del page_image_info[xref]["digest"]
    return page_image_info


# def get_page_text_info(page):
#     page_text_info_=[block for block in page.get_text("dict")["blocks"] if block["type"]==0]
#     page_text_info=[]
#     for page_text in page_text_info_:
#         for line in page_text["lines"]:
#             for span in line["spans"]:
#                 page_text_info.append(
#                     {
#                         "bbox": span["bbox"],
#                         "text": span["text"],
#                         "size": span["size"],
#                         "flags": span["flags"],
#                         "font": span["font"],
#                         "color": span["color"],
#                         "origin": span["origin"],
#                         "ascender": span["ascender"],
#                         "descender": span["descender"],
#                         "wmode": line["wmode"],
#                         "dir": line["dir"],
#                     }
#                 )

#     return page_text_info
def get_page_text_info(page):
    page_text_info_ = [block for block in page.get_text("dict")["blocks"] if block["type"] == 0]
    page_text_info = []
    for page_text in page_text_info_:
        text = ""
        for line in page_text["lines"]:
            for span in line["spans"]:
                text += span["text"] + " "
        text = text.strip()
        if text == "":
            continue
        page_text_info.append(
            {
                "bbox": page_text["bbox"],
                "text": text,
                "size": span["size"],
                "flags": span["flags"],
                "font": span["font"],
                "color": span["color"],
                "origin": span["origin"],
                "ascender": span["ascender"],
                "descender": span["descender"],
            }
        )

    return page_text_info


def extract(pdf_path, path_output_dir, use_pbar=True):
    pdf_path = Path(pdf_path)
    path_output_dir = Path(path_output_dir)

    if not os.path.exists(path_output_dir):  # make subfolder if needed
        os.mkdir(path_output_dir)
        print(f"Created output directory: {path_output_dir}")

    # shutil.copy2(path_meta_data.parent/param["path"], "/tmp/"+Path(param["path"]).name)
    # doc = fitz.open("/tmp/"+Path(param["path"]).name)
    doc = fitz.open(pdf_path)

    if use_pbar:
        pbar = tqdm(enumerate(doc, 1), total=len(doc))
    else:
        pbar = enumerate(doc, 1)

    image_total = 0
    image_saved = 0
    text_total = 0
    text_saved = 0
    for page_num, page in pbar:
        # per page operations
        # load image info from page
        page_image_info = get_page_image_info(page)

        # save image
        image_data_list = []
        page_image_info_retrieved = {}
        for xref, image_info in page_image_info.items():
            try:
                image_data = extract_fitz_pixmap(
                    doc=doc, xref=image_info["xref"], smask=image_info["smask"]
                )
            except ValueError:
                continue
            else:
                image_data_list.append(image_data)
                page_image_info_retrieved[xref] = image_info

        # load text info from page
        page_text_info = get_page_text_info(page)

        if len(page_image_info_retrieved) == 0:
            continue

        if not os.path.exists(path_output_dir / f"{page_num:05d}"):
            os.mkdir(path_output_dir / f"{page_num:05d}")

        # save text
        json.dump(
            page_text_info,
            open(path_output_dir / f"{page_num:05d}" / "text.json", "w"),
            indent=4,
            ensure_ascii=False,
        )

        # save image
        for image_data in image_data_list:
            img_filename = f'{image_data["xref"]:05d}.{image_data["ext"]}'
            with open(path_output_dir / f"{page_num:05d}" / img_filename, "wb") as f:
                f.write(image_data["image"])

        json.dump(
            page_image_info_retrieved,
            open(path_output_dir / f"{page_num:05d}" / "image.json", "w"),
            indent=4,
            ensure_ascii=False,
        )

        # record counts
        image_total += len(page_image_info)
        image_saved += len(page_image_info_retrieved)

        text_total += len(page_text_info)
        text_saved += len(page_text_info)
        if use_pbar:
            pbar.set_postfix(
                {
                    "image_total": image_total,
                    "image_saved": image_saved,
                    "text_total": text_total,
                    "text_saved": text_saved,
                }
            )

    doc.close()
    print(
        f"{pdf_path}: image_total: {image_total}, image_saved: {image_saved} text_total: {text_total}, text_saved: {text_saved}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pdf_extract.py",
        description="extract image and text from pdf files",
        epilog="This is a first preprocessing step for textbooks",
    )

    parser.add_argument("-i", "--input", type=str, help="input directory", required=True)
    parser.add_argument("-o", "--output", type=str, help="output directory", required=True)
    parser.add_argument("-t", "--thread", type=int, help="thread", required=False, default=1)

    # 140593310441360:
    # 'data/textbook/pdf_files/pdf_name.pdf'
    # 140593310509264:
    # 'data/textbook/pdf_files/pdf_name.pdf'
    # 140593310507728:
    # 'data/textbook/pdf_files/pdf_name.pdf'

    args = parser.parse_args()

    path_input_dir = Path(args.input)
    path_output_dir = Path(args.output)
    num_thread = args.thread

    pdf_path_list = sorted(glob.glob(str(path_input_dir / "*pdf")))

    if not os.path.exists(path_output_dir):  # make subfolder if needed
        os.mkdir(path_output_dir)
        print(f"Created output directory: {path_output_dir}")

    print(f"Number of pdf files: {len(pdf_path_list)}")
    if num_thread > 1:
        with Pool(processes=num_thread) as pool:
            pool.starmap(
                extract,
                [
                    [pdf_path, path_output_dir / Path(pdf_path).stem, False]
                    for pdf_path in pdf_path_list
                ],
            )
    else:
        for pdf_path in pdf_path_list:
            print(pdf_path)
            path_output_pdf_dir = path_output_dir / Path(pdf_path).stem
            extract(pdf_path=pdf_path, path_output_dir=path_output_pdf_dir, use_pbar=True)
