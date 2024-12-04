import requests
import tarfile
import os
import tqdm
import hashlib


CHECKSUMS={"rayonix": "2504ce9b87abbd9a721f641db16fdf40",
        "eiger": "9b9b77f113705a2389986633b68d55d5",
        "pilatus":"b9614eadfe6b1986bee4f873acd9f306"}


def dl(url):
    # https://stackoverflow.com/a/16696317/2077270
    # https://stackoverflow.com/a/37573701/2077270
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, ascii=True) as progress_bar:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
    return local_filename


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--md5", action="store_true", help="add this flag to verify file downloads with md5 checksum")
    args = ap.parse_args()

    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sims"))
    print("Downloading the simulation data:")
    data_folder = os.path.join(dirname, "for_tutorial")
    if not os.path.exists(data_folder):
        url='https://bl831.als.lbl.gov/~jamesh/resonet/for_tutorial.tar.gz'
        try:
            f = dl(url)
        except Exception as err:
            f = "for_tutorial.tar.gz"
            os.system(f"wget {url}")
        print("Opening simulation data and saving to %s:" % dirname)
        tar = tarfile.open(f, "r:gz")
        with tqdm.tqdm(total=len(tar.getmembers()), unit="files", unit_scale=True, ascii=True) as progress_bar:
            for i_member, member in enumerate(tar.getmembers()):
                progress_bar.update(1)
                tar.extract(member=member, path=dirname)
        print("Removing tar.gz archive.")
        os.remove(f)
    else:
        print(f"Warning, data folder {data_folder} exists, skipping re-download.")

    print("Downloading the image formats:")

    for name,ext in (("pilatus","cbf"), ("eiger","cbf"), ("rayonix", "mccd")):
        base_f = f"{name}_1_00001.{ext}"
        url = f"https://smb.slac.stanford.edu/~resonet/{base_f}"
        try:
            f=dl(url)
        except Exception as err:
            f=base_f
            os.system(f"wget {url}")

        if args.md5:
            md5 = hashlib.md5(open(f,'rb').read()).hexdigest()
            print(f"Checksum for {f}={md5}")
            assert md5==CHECKSUMS[name]
        name = os.path.join(dirname, os.path.basename(f))
        os.rename(f,name)
        print(f"... saved format to {name}")
    print("Done.")
    

if __name__=="__main__":
    main()

