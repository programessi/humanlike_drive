"""Resume the official archive with validated HTTP ranges and bounded retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

URL="https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("downloads/CARLA_0.9.15.tar.gz"))
    p.add_argument("--size",type=int,default=8386636048)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--direct",action="store_true",help="Connect directly instead of using environment proxy settings")
    args=p.parse_args()
    if args.direct:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    archive=args.output.resolve();archive.parent.mkdir(parents=True,exist_ok=True)
    parts=archive.parent/"carla_parts";parts.mkdir(exist_ok=True)
    metadata=parts/"download.json"
    prefix=parts/"prefix"
    if archive.exists() and archive.stat().st_size==args.size:
        print("Archive already complete",flush=True);return
    if not metadata.exists():
        base=archive.stat().st_size if archive.exists() else 0
        if base>args.size:raise RuntimeError("Existing archive exceeds official content length")
        if archive.exists():archive.rename(prefix)
        metadata.write_text(json.dumps({"url":URL,"size":args.size,"base":base}))
    state=json.loads(metadata.read_text())
    if state["size"]!=args.size:raise RuntimeError("Server content length changed")
    base=state["base"]
    chunk=64*1024*1024
    jobs=[(i,start,min(args.size-1,start+chunk-1)) for i,start in enumerate(range(base,args.size,chunk))]

    def download(job):
        index,start,end=job
        path=parts/f"{index:04d}.part"
        length=end-start+1
        for attempt in range(16):
            have=path.stat().st_size if path.exists() else 0
            if have==length:return index
            if have>length:raise RuntimeError(f"Oversized segment {index}")
            request=urllib.request.Request(URL,headers={"Range":f"bytes={start+have}-{end}","User-Agent":"humanlike-drive/0.1"})
            try:
                with urllib.request.urlopen(request,timeout=30) as response:
                    expected=f"bytes {start+have}-{end}/{args.size}"
                    if response.headers.get("Content-Range")!=expected:
                        raise RuntimeError(f"Invalid Content-Range for {index}: {response.headers.get('Content-Range')}")
                    with path.open("ab") as f:
                        while True:
                            block=response.read1(256*1024)
                            if not block:break
                            f.write(block)
                if path.stat().st_size==length:return index
            except Exception as error:
                if attempt==15:raise RuntimeError(f"Segment {index} failed") from error
                time.sleep(min(8,attempt+1))
        raise RuntimeError(f"Short segment {index}")

    last_report=0.;done=0
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(download,job) for job in jobs]):
            try:
                future.result();done+=1
            except Exception as error:
                failures.append(str(error));print(str(error),flush=True)
            now=time.monotonic()
            if now-last_report>15 or done==len(jobs):
                downloaded=base+sum(f.stat().st_size for f in parts.glob("*.part"))
                print(f"Downloaded {downloaded/1e9:.2f} / {args.size/1e9:.2f} GB ({done}/{len(jobs)} tail segments)",flush=True)
                last_report=now
    if failures:
        raise RuntimeError(f"Incomplete segments (rerun to resume): {failures}")
    assembled=archive.with_suffix(archive.suffix+".assembling")
    with assembled.open("wb") as out:
        if base:
            with prefix.open("rb") as source:
                remaining=base
                while remaining:
                    block=source.read(min(remaining,16*1024*1024))
                    if not block:raise RuntimeError("Download prefix became shorter")
                    out.write(block);remaining-=len(block)
        for index,_,_ in jobs:
            with (parts/f"{index:04d}.part").open("rb") as source:shutil.copyfileobj(source,out,16*1024*1024)
    if assembled.stat().st_size!=args.size:raise RuntimeError("Assembled length mismatch")
    os.replace(assembled,archive)
    print(f"Complete: {archive}",flush=True)


if __name__=="__main__":main()
