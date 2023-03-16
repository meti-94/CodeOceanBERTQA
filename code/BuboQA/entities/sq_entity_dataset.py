from torchtext import data

class SQdataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field, path,
               train='train.txt', validation='valid.txt', test='test.txt'):
        return super(SQdataset, cls).splits(
            path=path, train=train, validation=validation, test=test,
            format='tsv', fields=[('id', None), ('sub', None), ('entity', None), ('relation', None),
                                  ('obj', None), ('text', text_field), ('ed', label_field)]
        )
class ReverbDataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field, path,
               train='train.tsv', validation='valid.tsv', test='test.tsv'):
        return super(ReverbDataset, cls).splits(
            path=path, train=train, validation=validation, test=test,
            format='tsv', fields=[('text', text_field), ('ed', label_field)]
        )
