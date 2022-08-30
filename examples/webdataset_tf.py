import webdataset as wds
import torch
from torchvision import transforms

def identity(x):
    return x

class tf_webdataset:
    def __init__(self, batch_size):
        # num of your tar files
        url = "/COYO-700M/{000000..008191}.tar"
        
        preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(384),
            transforms.RandomResizedCrop(224),
        ])
      
        self.dataset = (
            wds.WebDataset(url)
            .shuffle(1000)
            .decode("rgb")
            .to_tuple("jpg", "txt")
            .map_tuple(preproc, identity)
        )

        self.dataset.batched(batch_size, partial=False)
        self.loader = (
            wds.WebLoader(self.dataset, num_workers=2, batch_size=None)
            .unbatched()
            .shuffle(1000)
        )
    
    def __iter__(self):
        for img, txt in self.loader:
            yield img, txt
    
    def output_types(self):
        return (tf.float32, tf.string)
    
    def output_shapes(self):
        return ((3, 224, 224), ())

def tf_webdataset_test(batch_size):
    dataset = tf_webdataset(batch_size)
    tdf = tf.data.Dataset.from_generator(
        generator=dataset.__iter__, output_types=dataset.output_types(), output_shapes=dataset.output_shapes()
    )
    tdf = tdf.batch(batch_size)
    tdf = tdf.prefetch(2)

    for step, (img, text) in enumerate(tdf):
        print(f'{step}: {text[0]}')

        if step == 32:
            break



def main():
    tf_webdataset_test(256)

if __name__ == '__main__':
    main()