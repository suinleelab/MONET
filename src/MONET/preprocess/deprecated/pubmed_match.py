import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from tqdm.contrib.concurrent import process_map

from MONET.utils.io import convert_dict, load_file_as_binary, load_hdf5, load_pkl


def process_article(article):

    if os.path.exists(article):
        soup = BeautifulSoup(load_file_as_binary(article), features="xml")
    elif isinstance(article, bytes):
        soup = BeautifulSoup(article, features="xml")
    else:
        raise ValueError(f"article must be a path or a byte, not {type(article)}. {article}")

    graphic_list = soup.find_all("graphic")

    graphic_record = []
    for graphic in graphic_list:
        fig_info = {}
        fig_info["href"] = graphic["xlink:href"]

        # parent checking
        parent = graphic.parent
        if parent.name == "alternatives":
            parent = parent.parent
        elif parent.name == "boxed-text":
            assert set(parent.attrs.keys()).issubset(
                {"position", "content-type", "id", "orientation"}
            ), parent.attrs.keys()
            parent = parent.parent
            assert parent.name == "p", f"Failed to locate valid parent for graphic: {graphic}"

            # if parent.name != "p":
            #     print(parent)
            #     raise

        if parent.name == "fig":
            # id
            if "id" in parent.attrs:
                fig_info["id"] = parent["id"]

            # label
            label = parent.find_all("label")
            fig_info["label"] = label

            # caption
            caption = parent.find_all("caption")
            fig_info["caption"] = caption
        elif parent.name == "p":
            fig_info["caption"] = parent
            # print("~~~~~p~~~~", article_id, parent.prettify())
        elif parent.name == "body":
            fig_info["caption"] = parent
            # print("~~~~~p~~~~", article_id, parent.prettify())
        elif parent.name == "abstract":
            fig_info["caption"] = parent
            # print("~~~~~p~~~~", article_id, parent.prettify())
        elif parent.name == "supplementary-material":
            if parent["content-type"] == "scanned-pages":
                continue

            assert (
                parent["content-type"] == "local-data"
            ), f"Unknown content-type: {parent['content-type']}"

            # id
            if "id" in parent.attrs:
                fig_info["id"] = parent["id"]

            # label
            label = parent.find_all("label")
            fig_info["label"] = label
            # if len(label)==1:
            #     fig_info['label']=label[0]
            # elif len(label)==0:
            #     pass
            # else:
            #     raise

            # caption
            caption = parent.find_all("caption")
            fig_info["caption"] = caption
        elif parent.name == "table-wrap":
            continue
        elif parent.name == "bio":
            continue
        elif parent.name == "disp-formula":
            continue
        elif parent.name == "td":
            # print("~~~~~td~~~~", article_id, parent.prettify())
            continue
        elif parent.name == "sec":
            # print("~~~~~sec~~~~", article_id, parent.prettify())
            continue
        elif parent.name == "floats-group":
            # print("~~~~~floats-group~~~~", article_id, parent.prettify())
            continue
        elif parent.name == "article":
            # print("~~~~~article~~~~", article_id, parent.prettify())
            continue
        elif parent.name == "inline-formula":
            # print("~~~~~inline-formula~~~~", article_id, parent.prettify())
            continue
        else:
            raise NotImplementedError(
                f"Unknown parent.name: {parent.name} \nparent: {parent.prettify()} \nparent.parent: {parent.parent.prettify()}"
            )

        if "label" in fig_info.keys():
            fig_info["label_text"] = [label.text for label in fig_info["label"]]
            fig_info["label"] = [str(label) for label in fig_info["label"]]
        else:
            fig_info["label_text"] = []
            fig_info["label"] = []

        if "caption" in fig_info.keys():
            fig_info["caption_text"] = [caption.text for caption in fig_info["caption"]]
            fig_info["caption"] = [str(caption) for caption in fig_info["caption"]]
        else:
            fig_info["caption_text"] = []
            fig_info["caption"] = []

        if "media_caption" in fig_info.keys():
            fig_info["media_caption_text"] = [
                caption.text for caption in fig_info["media_caption"]
            ]
            fig_info["media_caption"] = [str(caption) for caption in fig_info["media_caption"]]
        else:
            fig_info["media_caption_text"] = []
            fig_info["media_caption"] = []

        graphic_record.append(fig_info)

    media_list = soup.find_all("media")
    media_record = []
    for media in media_list:
        if os.path.splitext(media["xlink:href"])[1].lower() in [
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
        ]:
            continue

        fig_info = {}
        fig_info["href"] = media["xlink:href"]

        caption = media.find_all("caption")
        fig_info["media_caption"] = caption
        #         if len(caption)==1:
        #             fig_info['media_caption']=caption[0]
        #         elif len(caption)==0:
        #             pass
        #         else:
        #             print(parent.prettify())
        #             raise

        parent = media.parent
        if parent.name == "supplementary-material":
            # id
            if "id" in parent.attrs:
                fig_info["id"] = parent["id"]

            # label
            label = parent.find_all("label")
            fig_info["label"] = label
            #             if len(label)==1:
            #                 fig_info['label']=label[0]
            #             elif len(label)==0:
            #                 pass
            #             else:
            #                 raise

            # caption
            caption = parent.find_all("caption")
            fig_info["caption"] = caption

        if "label" in fig_info.keys():
            fig_info["label_text"] = [label.text for label in fig_info["label"]]
            fig_info["label"] = [str(label) for label in fig_info["label"]]
        else:
            fig_info["label_text"] = []
            fig_info["label"] = []

        if "caption" in fig_info.keys():
            fig_info["caption_text"] = [caption.text for caption in fig_info["caption"]]
            fig_info["caption"] = [str(caption) for caption in fig_info["caption"]]
        else:
            fig_info["caption_text"] = []
            fig_info["caption"] = []

        if "media_caption" in fig_info.keys():
            fig_info["media_caption_text"] = [
                caption.text for caption in fig_info["media_caption"]
            ]
            fig_info["media_caption"] = [str(caption) for caption in fig_info["media_caption"]]
        else:
            fig_info["media_caption_text"] = []
            fig_info["media_caption"] = []

        media_record.append(fig_info)

    return (graphic_record, media_record)


# process_article(('PMC1550589',article_binary_dict['PMC1550589']))
# process_article(('PMC2627840',article_binary_dict['PMC2627840']))
# process_article(('PMC1550589',article_binary_dict['PMC1550589']))
# cnt=0
# for key in np.random.choice(list(article_binary_dict.keys()), 1000, replace=False):
#     value=article_binary_dict[key]
#     cnt+=1
#     x=process_article((key,value))


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

    def process_article_wrapper(args):
        (article_id, article) = args
        graphic_record, media_record = process_article(article)

        for fig_info in graphic_record:
            fig_info["article_id"] = article_id
        for fig_info in media_record:
            fig_info["article_id"] = article_id
        return (graphic_record, media_record)

    # print(process_article_wrapper(next(iter(xml_dict_filtered.items()))))

    result = process_map(
        process_article_wrapper,
        xml_dict_filtered.items(),
        max_workers=16,
        chunksize=1000,
    )
    graphic_record = [fig_info for (graphic_record, _) in result for fig_info in graphic_record]
    media_record = [fig_info for (_, media_record) in result for fig_info in media_record]

    print(len(graphic_record), len(media_record))
    print(graphic_record[0], graphic_record[-1])
    print(media_record[0], media_record[-1])
    # for idx, data in enumerate(result):
    #     if idx < 10:
    #         print(data)
    # print(result)
