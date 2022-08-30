import webdataset as wds
import torch
from torchvision import transforms

def identity(x):
    return x

def dataloader_test():
    # num of your tar files
    url = "/COYO-700M/{000000..008191}.tar"
    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(384),
        transforms.CenterCrop(224),
    ])

    batch_size = 512

    dataset = (
        wds.WebDataset(url)
        .shuffle(1000)
        .decode("rgb")
        .to_tuple("jpg", "txt")
        .map_tuple(preproc, identity)
    )
		
    dataset.batched(batch_size, partial=False)
	
    loader = (
        wds.WebLoader(dataset, num_workers=2, batch_size=None)
    )

    for step, (img, text) in enumerate(loader):
        print(f'{step}: {text[0]}')

        if step == 32:
            break

def main():
    dataloader_test()

if __name__ == '__main__':
    main()