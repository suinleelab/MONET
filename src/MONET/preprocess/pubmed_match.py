import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from bs4 import BeautifulSoup
from tqdm.contrib.concurrent import process_map

from MONET.utils.io import convert_dict, load_file_as_binary, load_hdf5, load_pkl


def parse_graphic(graphic, verbose=False):
    fig_info = {}
    fig_info["href"] = graphic["xlink:href"]

    # checking parent tag name
    parent = graphic.parent
    if parent.name == "alternatives":
        parent = parent.parent
    elif parent.name == "boxed-text":
        assert set(parent.attrs.keys()).issubset(
            {"position", "content-type", "id", "orientation"}
        ), f"parent.attrs.keys(): {parent.attrs.keys()} should be subset of set(['position', 'content-type', 'id', 'orientation'])"
        parent = parent.parent
        assert parent.name == "p", f"Failed to locate valid parent for graphic: {graphic}"

    # finding labels and captions
    if parent.name == "fig":
        # assign id
        if "id" in parent.attrs:
            fig_info["id"] = parent["id"]

        # assign label
        label = parent.find_all("label")
        fig_info["label"] = label

        # assign caption
        caption = parent.find_all("caption")
        fig_info["caption"] = caption

        # print log
        if verbose:
            print(f"caption identified for {parent.name}", parent.prettify())
    elif parent.name == "p":
        # assign caption
        fig_info["caption"] = parent

        # print log
        if verbose:
            print(f"caption identified for {parent.name}", parent.prettify())
    elif parent.name == "body":
        # assign caption
        fig_info["caption"] = parent

        # print log
        if verbose:
            print(f"caption identified for {parent.name}", parent.prettify())
    elif parent.name == "abstract":
        fig_info["caption"] = parent

        # print log
        if verbose:
            print(f"caption identified for {parent.name}", parent.prettify())
    elif parent.name == "supplementary-material":
        if parent["content-type"] == "scanned-pages":
            return None
        assert (
            parent["content-type"] == "local-data"
        ), f"Unknown content-type: {parent['content-type']}"

        # assign id
        if "id" in parent.attrs:
            fig_info["id"] = parent["id"]

        # assign label
        label = parent.find_all("label")
        fig_info["label"] = label
        # if len(label)==1:
        #     fig_info['label']=label[0]
        # elif len(label)==0:
        #     pass
        # else:
        #     raise

        # assign caption
        caption = parent.find_all("caption")
        fig_info["caption"] = caption

        # print log
        if verbose:
            print(f"caption identified for {parent.name}", parent.prettify())
    elif parent.name == "table-wrap":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "bio":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "disp-formula":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "td":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "sec":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "floats-group":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "article":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    elif parent.name == "inline-formula":
        # print log
        if verbose:
            print(f"caption not identified for {parent.name}", parent.prettify())
        return None
    else:
        raise NotImplementedError(
            f"Unknown parent.name: {parent.name} \nparent: {parent.prettify()} \nparent.parent: {parent.parent.prettify()}"
        )

    # extracting text from labels and captions
    if "label" in fig_info.keys():
        fig_info["label_text"] = [label.text for label in fig_info["label"]]
        fig_info["label_str"] = [str(label) for label in fig_info["label"]]
        del fig_info["label"]
    else:
        fig_info["label_text"] = []
        fig_info["label_str"] = []

    if "caption" in fig_info.keys():
        fig_info["caption_text"] = [caption.text for caption in fig_info["caption"]]
        fig_info["caption_str"] = [str(caption) for caption in fig_info["caption"]]
        del fig_info["caption"]
    else:
        fig_info["caption_text"] = []
        fig_info["caption_str"] = []

    if "media_caption" in fig_info.keys():
        fig_info["media_caption_text"] = [caption.text for caption in fig_info["media_caption"]]
        fig_info["media_caption_text"] = [str(caption) for caption in fig_info["media_caption"]]
        del fig_info["media_caption"]
    else:
        fig_info["media_caption_text"] = []
        fig_info["media_caption"] = []
    return fig_info


def parse_media(
    media,
):

    ext_valid = [
        ".doc",
        ".docx",
        ".pdf",
        ".zip",
        ".xlsx",
        ".xls",
        ".wmv",
        ".rar",
        ".pptx",
        ".avi",
        ".txt",
        ".mp4",
        ".ppt",
        ".ai",
        ".ods",
        ".csv",
        ".html",
        ".fas",
        ".mov",
        ".tgz",
        ".xml",
        ".htm",
        ".eps",
        ".sav",
        ".tsv",
        ".fasta",
        ".fa",
        ".ppsx",
        ".bz2",
        ".psd",
        ".rtf",
        ".tab",
    ]

    if os.path.splitext(media["xlink:href"])[1].lower() in ext_valid:
        return None

    fig_info = {}
    fig_info["href"] = media["xlink:href"]

    caption = media.find_all("caption")
    fig_info["media_caption"] = caption
    # if len(caption)==1:
    #     fig_info['media_caption']=caption[0]
    # elif len(caption)==0:
    #     pass
    # else:
    #     print(parent.prettify())
    #     raise

    parent = media.parent
    if parent.name == "supplementary-material":
        # assign id
        if "id" in parent.attrs:
            fig_info["id"] = parent["id"]

        # assign label
        label = parent.find_all("label")
        fig_info["label"] = label
        # if len(label)==1:
        #     fig_info['label']=label[0]
        # elif len(label)==0:
        #     pass
        # else:
        #     raise

        # assign caption
        caption = parent.find_all("caption")
        fig_info["caption"] = caption

    # extracting text from labels and captions
    if "label" in fig_info.keys():
        fig_info["label_text"] = [label.text for label in fig_info["label"]]
        fig_info["label_str"] = [str(label) for label in fig_info["label"]]
        del fig_info["label"]
    else:
        fig_info["label_text"] = []
        fig_info["label_str"] = []

    if "caption" in fig_info.keys():
        fig_info["caption_text"] = [caption.text for caption in fig_info["caption"]]
        fig_info["caption_str"] = [str(caption) for caption in fig_info["caption"]]
        del fig_info["caption"]
    else:
        fig_info["caption_text"] = []
        fig_info["caption_str"] = []

    if "media_caption" in fig_info.keys():
        fig_info["media_caption_text"] = [caption.text for caption in fig_info["media_caption"]]
        fig_info["media_caption_str"] = [str(caption) for caption in fig_info["media_caption"]]
        del fig_info["media_caption"]
    else:
        fig_info["media_caption_text"] = []
        fig_info["media_caption_str"] = []
    return fig_info


def parse_pubmedxml(article):
    if os.path.exists(article):
        soup = BeautifulSoup(load_file_as_binary(article), features="xml")
    elif isinstance(article, bytes):
        soup = BeautifulSoup(article, features="xml")
    else:
        raise ValueError(f"article must be a path or a byte, not {type(article)}. {article}")

    graphic_list = soup.find_all("graphic")
    graphic_record = []
    for graphic in graphic_list:
        fig_info = parse_graphic(graphic)
        if fig_info is not None:
            graphic_record.append(fig_info)

    media_list = soup.find_all("media")
    media_record = []
    for media in media_list:
        fig_info = parse_media(media)
        if fig_info is not None:
            media_record.append(fig_info)

    # return (article_id, graphic_record, media_record)
    return (graphic_record, media_record)


def clean_caption(caption):
    caption = re.sub(r"Supplementary material [0-9]", "", caption)
    caption = re.sub(r"Supplementary material [0-9]", "", caption)
    caption = re.sub(
        r"Additional file [0-9]: Supplementary Figure [0-9].", "", caption
    )  # Additional file 1: Fig. S1
    caption = re.sub(r"Additional file [0-9]: Supplementary Figure S[0-9].", "", caption)
    caption = re.sub(r"Additional file [0-9]: Figure S[0-9].", "", caption)
    caption = re.sub(r"Additional file [0-9]:\xa0Fig. S[0-9].", "", caption)

    caption = caption.strip()
    if caption == "Click here for additional data file.":
        return None
    if caption == "Click here for file":
        return None
    elif caption == "\n":
        return None
    elif caption == "":
        return None
    elif caption == "Supplementary data":
        return None
    elif caption == "®":
        return None
    elif caption == ".":
        return None
    elif caption == "etc":
        return None
    elif re.match(r"Authors’ original file for figure [0-9]", caption) is not None:
        return None
    elif re.match(r"Authors' original file for figure [0-9]", caption) is not None:
        return None
    else:
        return caption
    # caption_list_new.append(caption)
    # return caption_list_new


def add_caption_final(fig_info):

    caption_cleaned_list = []
    for caption in fig_info["caption_text"]:
        caption_cleaned = clean_caption(caption)
        if caption_cleaned is not None:
            caption_cleaned_list.append(caption_cleaned)
    fig_info["caption_text_cleaned"] = caption_cleaned_list

    caption_cleaned_list = []
    for caption in fig_info["media_caption_text"]:
        caption_cleaned = clean_caption(caption)
        if caption_cleaned is not None:
            caption_cleaned_list.append(caption_cleaned)
    fig_info["media_caption_text_cleaned"] = caption_cleaned_list

    caption_final = list(
        OrderedDict.fromkeys(
            fig_info["caption_text_cleaned"] + fig_info["media_caption_text_cleaned"]
        )
    )
    fig_info["caption_final"] = caption_final
    return fig_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="pubmed_match.py",
        description="pubmed match",
        epilog="",
    )

    parser.add_argument("--image", type=str, help="image path", required=True)
    parser.add_argument("--xml", type=str, help="xml path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_image = Path(args.image)
    path_xml = Path(args.xml)
    path_output = Path(args.output)

    print("Loading image dataset...")

    if path_image.suffix == ".hdf5":
        image_dict = load_hdf5(path_input=path_image, field="images", verbose=True)
    elif path_image.suffix == ".pkl":
        image_dict = load_pkl(path_input=path_image, field="images", verbose=True)
    else:
        raise ValueError(f"Unknown file type: {path_image.suffix}")

    print("Loading xml dataset...")
    if path_xml.suffix == ".hdf5":
        xml_dict = load_hdf5(path_input=path_xml, field="xml", verbose=True)
    elif path_xml.suffix == ".pkl":
        xml_dict = load_pkl(path_input=path_xml, field="xml", verbose=True)
    else:
        raise ValueError(f"Unknown file type: {path_xml.suffix}")

    image_article_key = {"_".join(key.split("_")[:7]) for key in image_dict.keys()}
    xml_article_key = {"_".join(key.split("_")[:7]) for key in xml_dict.keys()}
    assert image_article_key.issubset(xml_article_key), "Image article not in xml article"

    xml_dict_filtered = OrderedDict()
    for key in xml_dict.keys():
        if "_".join(key.split("_")[:7]) in image_article_key:
            xml_dict_filtered[key] = xml_dict[key]

    print("Converting to binary...")

    print(len(image_dict), len(xml_dict))
    print(len(image_article_key), len(xml_article_key), len(xml_dict_filtered))

    print(np.unique([os.path.splitext(key)[1] for key in xml_dict.keys()], return_counts=True))
    print("Matching xml and image...")

    def parse_pubmedxml_wrapper(args):
        (article_id, article) = args
        graphic_record, media_record = parse_pubmedxml(article)

        for fig_info in graphic_record:
            fig_info["article_id"] = article_id
        for fig_info in media_record:
            fig_info["article_id"] = article_id
        return (graphic_record, media_record)

    xml_parsed = process_map(
        parse_pubmedxml_wrapper,
        xml_dict_filtered.items(),
        max_workers=16,
        chunksize=100,
    )  # print(process_article_wrapper(next(iter(xml_dict_filtered.items()))))
    graphic_record = [
        fig_info for (graphic_record, _) in xml_parsed for fig_info in graphic_record
    ]
    media_record = [fig_info for (_, media_record) in xml_parsed for fig_info in media_record]

    # process_fig_info(graphic_record[0])

    print(len(graphic_record), len(media_record))
    # print(graphic_record[0], graphic_record[-1])
    # print(media_record[0], media_record[-1])

    fig_info_final = process_map(
        add_caption_final,
        graphic_record + media_record,
        max_workers=16,
        chunksize=100,
    )

    fig_info_final_df = pd.DataFrame(
        fig_info_final,
        columns=[
            "href",
            "id",
            "article_id",
            "label_text",
            "caption_final",
        ],
    )
    print(f"Before removing empty: {len(fig_info_final_df)}")
    fig_info_final_df = fig_info_final_df[
        fig_info_final_df["caption_final"].map(lambda x: len(x) > 0)
    ]
    print(f"Before removing empty: {len(fig_info_final_df)}")
    fig_info_final_df["caption_final_str"] = fig_info_final_df["caption_final"].astype(str)

    print(f"Before dropping duplicated {len(fig_info_final_df)}")
    fig_info_final_df = fig_info_final_df[
        ~fig_info_final_df.duplicated(subset=["article_id", "href", "caption_final_str"])
    ]
    print(f"After dropping duplicated {len(fig_info_final_df)}")

    image_key_list = [os.path.splitext(key)[0] for key in image_dict.keys()]
    assert len(set(image_key_list)) == len(image_dict.keys()), "Duplicate image key"

    fig_info_final_df["image_key_1"] = fig_info_final_df.apply(
        lambda x: "_".join(x["article_id"].split("_")[:7]) + "_" + x["href"], axis=1
    )
    fig_info_final_df["image_key_2"] = fig_info_final_df.apply(
        lambda x: "_".join(x["article_id"].split("_")[:7]) + "_" + os.path.splitext(x["href"])[0],
        axis=1,
    )
    fig_info_final_df_all = pd.concat(
        [
            fig_info_final_df.set_index("image_key_1").loc[
                pd.Index(image_key_list).intersection(
                    fig_info_final_df.set_index("image_key_1").index
                )
            ],
            fig_info_final_df.set_index("image_key_2").loc[
                pd.Index(image_key_list).intersection(
                    fig_info_final_df.set_index("image_key_2").index
                )
            ],
        ]
    ).drop(columns=["image_key_1", "image_key_2"])
    fig_info_final_df_all = fig_info_final_df_all.reset_index()

    fig_info_final_df_all = fig_info_final_df_all[
        ~fig_info_final_df_all.duplicated(
            subset=["index", "article_id", "href", "caption_final_str"]
        )
    ].set_index("index")
    fig_info_final_df_all = fig_info_final_df_all.drop(columns=["caption_final_str"])

    print(f"Final: {len(fig_info_final_df_all)}")

    if path_output.suffix == ".pkl":
        fig_info_final_df_all.to_pickle(path_output)
    else:
        raise ValueError(f"Unknown file type: {path_output.suffix}")
