import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import tqdm
import yaml

from MONET.utils.io import get_hdf5_key, load_pkl


def process_text(text_raw):
    text_processed = (
        text_raw.strip().replace("\n", " ").replace("\t", " ").replace("•", " ").replace("�", " ")
    )
    text_processed = text_processed.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    text_processed = text_processed.lower().strip()
    # print(text_processed)

    if text_processed.startswith("figure"):
        text_processed = re.sub(r"\Afigure", "", text_processed)
    elif text_processed.startswith("efig"):
        text_processed = re.sub(r"\Aefig", "", text_processed)
    elif text_processed.startswith("fig"):
        text_processed = re.sub(r"\Afig", "", text_processed)

    text_processed = text_processed.strip()
    text_processed = re.sub(r"\A[\-\.\s0-9\:]+", "", text_processed)
    # text_processed = text_processed.split("(courtesy")[0]
    text_processed = text_processed.strip()

    return text_processed


def text_remove_legend(text_raw):
    for color in ["black ", "yellow ", "red ", "white ", "gay ", ""]:
        for shape in [
            "arrow",
            "arrows",
            "box",
            "boxes",
            "clrcle",
            "circle",
            "circles",
            "star",
            "stars",
            "dahsed line",
            "dahsed lines",
            "dotted line",
            "dotted lines",
        ]:
            for left_paren in ["(", "{"]:
                for right_paren in [")", "}"]:
                    text_raw = text_raw.replace(f"{left_paren}{color}{shape}{right_paren}", "")
    if text_raw.count("(") == 1:
        text_raw = text_raw.split("(")[0]
    return text_raw


def match_text(
    path_base,
    key_images,
    text_include_list,
    fontsize_range,
    font_list,
    prioritize_text_under_image,
    return_all=False,
    verbose=False,
):
    pbar = tqdm.tqdm(key_images)

    image_matched = 0
    image_skipped = 0

    text_matched = []
    for key in pbar:

        pbar.set_postfix(
            {
                "image_matched": image_matched,
                "image_skipped": image_skipped,
            }
        )

        pdf_name, page_num, xref = os.path.splitext(key)[0].split("_")

        # if not (pdf_name == "pdf_name" and page_num == "00025"):
        # if not (pdf_name == "pdf_name"):
        #     continue

        image_info_df = pd.DataFrame(
            json.load(open(path_base / pdf_name / page_num / "image.json"))
        ).T
        image_info_df.index = image_info_df.index.astype(int)
        image_info = image_info_df.loc[int(xref)]

        text_info_df = pd.DataFrame(json.load(open(path_base / pdf_name / page_num / "text.json")))
        text_info_df["image_key"] = key
        text_info_df["image_pdf_name"] = pdf_name
        text_info_df["image_page_num"] = page_num
        text_info_df["image_xref"] = xref

        # print("before filtering\n", text_info_df)
        # for i in text_info_df["text"]:
        #     print(i)
        if verbose:
            print(text_info_df)
        if len(text_info_df) == 0:
            print(pdf_name, page_num, xref, "no text")
            image_skipped += 1
            continue
        if text_include_list is not None:
            text_info_df = text_info_df[
                text_info_df["text"].apply(
                    lambda x: any(
                        [
                            all([t in x.lower() for t in text_include])
                            for text_include in text_include_list
                        ]
                    )
                )
            ]
        if verbose:
            print(f"after filtering text: {text_info_df}")
        if len(text_info_df) == 0:
            print(pdf_name, page_num, xref, "no text after filtering text")
            image_skipped += 1
            continue

        if fontsize_range is not None:
            text_info_df = text_info_df[
                text_info_df["size"].apply(
                    lambda x: x >= fontsize_range[0] and x <= fontsize_range[1]
                )
            ]
        if verbose:
            print(f"after filtering fontsize: {text_info_df}")
        if len(text_info_df) == 0:
            print(pdf_name, page_num, xref, "no text after filtering fontsize")
            image_skipped += 1
            continue

        if font_list is not None:
            text_info_df = text_info_df[
                text_info_df["font"].apply(lambda x: any([font == x for font in font_list]))
            ]
        if verbose:
            print(f"after filtering font: {text_info_df}")
        if len(text_info_df) == 0:
            print(pdf_name, page_num, xref, "no text after filtering font")
            image_skipped += 1
            continue

        if not isinstance(image_info["bbox"], list):
            print(pdf_name, page_num, xref, "image bbox is not list")
            image_skipped += 1
            continue
        text_info_df["edge_dist_x"] = text_info_df["bbox"].apply(
            lambda x: min(
                abs(x[0] - image_info["bbox"][0]),
                abs(x[0] - image_info["bbox"][2]),
                abs(x[2] - image_info["bbox"][0]),
                abs(x[2] - image_info["bbox"][2]),
            )
        )

        text_info_df["edge_dist_y"] = text_info_df["bbox"].apply(
            lambda x: min(
                abs(x[1] - image_info["bbox"][1]),
                abs(x[1] - image_info["bbox"][3]),
                abs(x[3] - image_info["bbox"][1]),
                abs(x[3] - image_info["bbox"][3]),
            )
        )

        text_info_df["edge_dist"] = (
            text_info_df["edge_dist_x"] ** 2 + text_info_df["edge_dist_y"] ** 2
        ) ** (0.5)

        text_info_df["text_under_image"] = text_info_df["bbox"].apply(
            lambda x: (x[1] + x[3]) >= (image_info["bbox"][1] + image_info["bbox"][3])
        )

        # image_center=(image_bbox[0]+image_bbox[2])/2, (image_bbox[1]+image_bbox[3])/2
        # for block_id, text_block in page_text_blocks.items():
        #     text_bbox=text_block["bbox"]
        #     text_block_center=(text_bbox[0]+text_bbox[2])/2, (text_bbox[1]+text_bbox[3])/2
        if return_all:
            text_info_df["text_formatted"] = (
                text_info_df["text"].apply(process_text).apply(text_remove_legend)
            )
        else:
            text_info_df["text_formatted"] = text_info_df["text"].apply(process_text)

        text_info = text_info_df.sort_values("edge_dist", ascending=True)

        if prioritize_text_under_image is not None:
            if prioritize_text_under_image:
                text_info = text_info.sort_values("text_under_image", ascending=False)
            else:
                text_info = text_info.sort_values("text_under_image", ascending=True)
        # print(text_info_df)
        if return_all:
            for _, row in text_info.iterrows():
                text_matched.append(row)
        else:
            text_matched.append(text_info.iloc[0])

        # print(text_info_df.sort_values("edge_dist").head(10))
        image_matched += 1
    text_matched = pd.DataFrame(text_matched)
    return text_matched


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="pdf_match.py",
        description="pdf match",
        epilog="",
    )

    parser.add_argument("--image", type=str, help="image path", required=True)
    parser.add_argument("--pdf-extracted", type=str, help="pdf extracted path", required=True)
    parser.add_argument("--config", type=str, help="pdf extracted path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_image = Path(args.image)
    path_pdf_extracted = Path(args.pdf_extracted)
    path_output = Path(args.output)
    path_config = Path(args.config)

    # data_dict = load_hdf5(path_input=path_image, field="images", verbose=True)
    if path_image.suffix == ".hdf5":
        key_images = get_hdf5_key(path_input=path_image, field="images")
    elif path_image.suffix == ".pkl":
        key_images = load_pkl(path_input=path_image, field="images")

    with open(path_config) as f:
        config = json.load(f)

    # config = {key: config[key] for key in list(config.keys())[::-1]}

    text_matched_all = []
    # check = True
    for pdf_name, pdf_config_list in config.items():

        # if check and pdf_name != "pdf_name":
        #     continue
        # else:
        #     check = False

        for pdf_config in pdf_config_list:
            print(pdf_name, pdf_config)
            text_include_list = pdf_config["text_include_list"]
            fontsize_range = pdf_config["fontsize_range"]
            font_list = pdf_config["font_list"]
            prioritize_text_under_image = pdf_config["prioritize_text_under_image"]
            return_all = pdf_config["return_all"]
            key_images_pdf = [
                key
                for key in key_images
                if os.path.splitext(key)[0].split("_")[0] == pdf_name
                # and os.path.splitext(key)[0].split("_")[1] == "00742"
            ]
            # print(key_images_pdf)

            text_matched = match_text(
                path_base=path_pdf_extracted,
                key_images=key_images_pdf,
                text_include_list=text_include_list,
                fontsize_range=fontsize_range,
                font_list=font_list,
                prioritize_text_under_image=prioritize_text_under_image,
                return_all=return_all,
                verbose=False,
            )

            text_matched_all.append(text_matched)

            # AdvP42D087   8.468
            # text_matched[text_matched["text_formatted"].str.contains("nodul")]
            # print(text_matched)
            # print(text_matched["image_key"])
            # print("size","\n", text_matched["size"].value_counts())
            # print(
            #     "edge_dist","\n",
            #     text_matched["edge_dist"].value_counts(
            #         bins=[0, 5, 10, 15, 20, 25, 30, 35]
            #     ),
            # )

    text_matched_all_df = pd.concat(text_matched_all)

    # text_matched_all_df = text_matched_all_df[
    #     ~text_matched_all_df.duplicated(subset=["image_key", "text_formatted"])
    # ]

    if path_output.suffix == ".csv":
        text_matched_all_df.to_csv(path_output)
    elif path_output.suffix == ".pkl":
        text_matched_all_df.to_pickle(path_output)
    else:
        raise ValueError("output file type not supported")
