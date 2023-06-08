import argparse
import ftplib
import os
import shutil
import tarfile
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
import tqdm
from tqdm.contrib.concurrent import process_map


def get_oa_list(search_csv, term_list=["derm", "skin"]):

    oa_file_list = pd.read_csv(
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv", dtype={"PMID": str}
    )  # PMC always 4,970,445

    oa_file_list_PMID = oa_file_list.set_index("PMID").loc[
        list(set(oa_file_list["PMID"]).intersection(set(search_csv["PMID"])))
    ]  # 471,717
    oa_file_list_PMCID = oa_file_list.set_index("Accession ID").loc[
        list(set(oa_file_list["Accession ID"]).intersection(set(search_csv["PMCID"])))
    ]  # 471,717

    oa_file_list_search = pd.concat(
        [oa_file_list_PMID.reset_index(), oa_file_list_PMCID.reset_index()]
    )
    oa_file_list_search = oa_file_list_search.drop_duplicates(subset="File", keep="first")

    oa_file_list_journal = oa_file_list[
        oa_file_list["Article Citation"].str.lower().str.contains(term_list[0])
        | oa_file_list["Article Citation"].str.lower().str.contains(term_list[1])
    ]

    oa_file_list_concat = pd.concat([oa_file_list_search, oa_file_list_journal])

    oa_file_list_concat_nondup = oa_file_list_concat.drop_duplicates(subset="File", keep="first")

    return oa_file_list_concat_nondup


def setup_ftp_connection(hostname):
    ftp = ftplib.FTP()
    ftp.connect(hostname)
    ftp.login("anonymous", "anonymous@domain.com")
    ftp.set_pasv(True)
    time.sleep(2)
    return ftp


def check_and_setup_ftp_connection(hostname, ftp=None):
    if ftp is None:
        ftp = setup_ftp_connection(hostname)
        ftp.voidcmd("NOOP")
        return ftp
    else:
        try:
            ftp.voidcmd("NOOP")
        except BaseException as e:
            ftp = setup_ftp_connection(hostname)
            ftp.voidcmd("NOOP")
        return ftp


def extract_tar(tar_path, output_dir, include_extension_list):
    # output_dir=os.path.dirname(local_path)
    # include_extension_list=[[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp"],
    #                         [".xml",".nxml"]
    #                         ]

    with tarfile.open(tar_path, "r:gz") as tar:
        for tarinfo in tar:
            if not tarinfo.isfile():
                tar.extract(tarinfo, output_dir)

        for include_extension in include_extension_list:
            ext_tarinfo_dict = OrderedDict()
            for tarinfo in tar:
                if os.path.splitext(tarinfo.name)[1] in include_extension:
                    ext_tarinfo_dict.setdefault(os.path.splitext(tarinfo.name)[0], []).append(
                        tarinfo
                    )

            ext_tarinfo_dict = OrderedDict(
                {
                    key: sorted(
                        value,
                        key=lambda x: include_extension.index(os.path.splitext(x.name)[1]),
                    )
                    for key, value in ext_tarinfo_dict.items()
                }
            )

            for key, value in ext_tarinfo_dict.items():
                tar.extract(value[0], output_dir)


def download_and_extract_article(
    file_list,
    hostname,
    local_dir,
    include_extension_list=[],
    blocksize=33554,
    use_pbar=True,
):
    local_dir = Path(local_dir)

    ftp = check_and_setup_ftp_connection(hostname)
    if use_pbar:
        pbar = tqdm.tqdm(file_list)
    else:
        pbar = file_list

    num_downloaded = 0
    num_skipped = 0
    num_error = 0

    for file in pbar:
        if use_pbar:
            pbar.set_postfix(
                {
                    "downloaded": num_downloaded,
                    "skipped": num_skipped,
                    "error": num_error,
                }
            )

        local_path = local_dir / file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(str(local_path).replace(".tar.gz", "")):
            num_skipped += 1
            continue

        for _ in range(5):
            ftp = check_and_setup_ftp_connection(hostname, ftp=ftp)
            try:
                with open(str(local_path) + ".download", "w+b") as f:
                    res = ftp.retrbinary("RETR %s" % file, f.write, blocksize=blocksize)
            except BaseException as e:
                print("error", e)
                if os.path.exists(str(local_path) + ".download"):
                    os.remove(str(local_path) + ".download")
            else:
                if res.startswith("226 Transfer complete"):
                    os.rename(str(local_path) + ".download", local_path)
                    break
                else:
                    print(f"Downloaded of file {file} is not complete.")
                    os.remove(str(local_path) + ".download")
            # except KeyboardInterrupt as e:
            #     os.remove(str(local_path) + ".download")
            #     raise e

        # check if file is downloaded
        assert os.path.exists(local_path), f"File {file} was not downloaded."
        assert "".join(local_path.suffixes) == ".tar.gz", f"File {file} is not tar.gz file."

        # extract files
        try:
            extract_tar(
                local_path,
                output_dir=os.path.dirname(local_path),
                include_extension_list=include_extension_list,
            )

        except BaseException as e:
            shutil.rmtree(str(local_path).replace(".tar.gz", ""))
            os.remove(local_path)
            num_error += 1
            # remove tar file
        else:
            os.remove(local_path)
            num_downloaded += 1
            # remove tar file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pubmed_download.py",
        description="download pubmed data",
        epilog="",
    )

    sub_parsers = parser.add_subparsers(dest="cmd")
    parser_filter = sub_parsers.add_parser("filter", help="filter")

    parser_filter.add_argument("-i", "--input", type=str, help="input path", required=True)
    parser_filter.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser_download = sub_parsers.add_parser("download", help="download")

    parser_download.add_argument("-i", "--input", type=str, help="input path", required=True)

    parser_download.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser_download.add_argument(
        "-e",
        "--extension",
        nargs="+",
        help="extract files with these extensions",
        required=True,
    )

    parser_download.add_argument(
        "-t", "--thread", type=int, help="thread", required=False, default=1
    )
    parser_download.add_argument("--index", type=int, help="index", required=False, default=1)

    parser_download.add_argument("--modulo", type=int, help="modulo", required=False, default=1)

    args = parser.parse_args()

    if args.cmd == "filter":
        print("Arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        path_input = Path(args.input)
        path_output = Path(args.output)
        search_csv = pd.read_csv(path_input, dtype={"PMID": str})  # PMID always 1,068,357
        oa_file_list_final = get_oa_list(search_csv, term_list=["derm", "skin"])
        oa_file_list_final.to_csv(path_output, index=False)

    elif args.cmd == "download":
        print("Arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        path_input = Path(args.input)
        path_output = Path(args.output)
        include_extension_list = args.extension
        num_thread = args.thread
        index = args.index
        modulo = args.modulo

        oa_file_list = pd.read_csv(path_input, index_col=0)
        include_extension_list = [
            include_extension.split(",") for include_extension in include_extension_list
        ]
        print(f"extension to extract: {include_extension_list}")

        file_list = ("pub/pmc/" + oa_file_list["File"]).tolist()
        if modulo > 1:
            file_list = [file for idx, file in enumerate(file_list) if idx % modulo == index]

        if num_thread > 1:
            file_list_split = [
                file_list[i * (16) : min((i + 1) * (16), len(file_list))]
                for i in range(len(file_list) // 16 + 1)
            ]

            download_and_extract_article_partial = partial(
                download_and_extract_article,
                hostname="ftp.ncbi.nlm.nih.gov",
                local_dir=path_output,
                include_extension_list=include_extension_list,
                blocksize=33554,
                use_pbar=False,
            )

            # download_and_extract_article_partial(file_list_split[0])

            process_map(
                download_and_extract_article_partial,
                file_list_split,
                max_workers=num_thread,
            )

        else:
            download_and_extract_article(
                hostname="ftp.ncbi.nlm.nih.gov",
                file_list=file_list,
                local_dir=path_output,
                include_extension_list=include_extension_list,
                blocksize=33554,
                use_pbar=True,
            )
    else:
        raise NotImplementedError(f"Command {args.cmd} not implemented")
