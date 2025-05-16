from . import RNASeq

def get_datasets(args):
    if args.dataset_type == 'rnaseq':
        return RNASeq.build(args)

    raise NotImplementedError
