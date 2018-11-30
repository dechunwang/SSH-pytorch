from model.dataset.wider import wider

__sets = {}

for split in ['train','val','test']:
    name = 'wider_{}'.format(split)
    __sets[name] = (lambda split=split: wider(split))


    def get_imdb(name):
        """Get an imdb (image database) by name."""
        try:
            __sets[name]
        except KeyError:
            raise KeyError('Unknown dataset: {}'.format(name))
        return __sets[name]()