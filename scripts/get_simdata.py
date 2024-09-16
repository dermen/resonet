import requests
import tarfile
import os
import tqdm


def dl(url):
    # https://stackoverflow.com/a/16696317/2077270
    # https://stackoverflow.com/a/37573701/2077270
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    #with requests.get(url, stream=True) as r:
    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, ascii=True) as progress_bar:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
    return local_filename


def main():
    dirname = os.path.join(os.path.dirname(__file__), "../sims")
    print("Downloading the simulation data:")
    url='https://bl831.als.lbl.gov/~jamesh/resonet/for_tutorial.tar.gz'
    f = dl(url)
    print("Opening simulation data and saving to %s:" % dirname)
    tar = tarfile.open(f, "r:gz")
    with tqdm.tqdm(total=len(tar.getmembers()), unit="files", unit_scale=True, ascii=True) as progress_bar:
        for i_member, member in enumerate(tar.getmembers()):
            progress_bar.update(1)
            tar.extract(member=member, path=dirname)
    print("Removing tar.gz archive.")
    os.remove(f)
    print("Done.")
    

if __name__=="__main__":
    main()

