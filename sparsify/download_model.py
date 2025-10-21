
import os
import requests
import tqdm
import hashlib

CHECKSUMS = {"1k_randoms.compress_3_withName.out":
        ("https://smb.slac.stanford.edu/~dermen/sparsify/{name}", "d0310995c875b4258cf4839f3429ab56")}


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
    print("Downloading model file")
    for name in CHECKSUMS:
        url_template, checksum = CHECKSUMS[name]
        url= url_template.format(name=name)
        try:
            f = dl(url)
        except Exception as err:
            os.system(f"wget {url}")
            f=name
        md5 = hashlib.md5(open(f, 'rb').read()).hexdigest()
        print(f"Checksum for {f}={md5}")
        assert md5 == checksum
        dirname = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        download_dir = os.path.join(dirname, "../downloaded_models")
        os.makedirs(download_dir, exist_ok=True)
        name = os.path.join( download_dir, os.path.basename(f))
        os.rename(f, name)
        print(f"Model saved to {name}.")

if __name__=="__main__":
    main()
