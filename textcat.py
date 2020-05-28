#!/usr/bin/env python3
"""
Computes the log probability of the sequence of tokens in file,
according to a trigram model.  The training source is specified by
the currently open corpus, and the smoothing method used by
prob() is polymorphic.
"""
import argparse
import logging
from pathlib import Path
import math
import numpy as np
try:
    # Numpy is your friend. Not *using* it will make your program so slow.
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import numpy as np
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


from Probs import LanguageModel

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, train_file: Path) -> Path:
    return Path(f"{smoother}_{lexicon.name}_{train_file.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "smoother",
        type=str,
        help="""Possible values: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the "1" in add1/backoff_add1 can be replaced with any real λ >= 0
   the "1" in loglinear1 can be replaced with any C >= 0 )
""",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word vector file; only used in the loglinear model",
    )
    #parser.add_argument("train_file", type=Path, help="location of the training corpus")
    parser.add_argument("train_file1", type=Path, help="location of the training corpus 1 for text categorization problem")
    parser.add_argument("train_file2", type=Path, help="location of the training corpus 2 for text categorization problem")
    parser.add_argument("gen_prior", type=float, help="prior prob that a test file is gen", nargs="?")
    parser.add_argument("test_files", type=Path, nargs="*")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()

    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and (not args.test_files or not args.gen_prior):
        parser.error("No test files or no gen prior specified.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path1 = get_model_filename(args.smoother, args.lexicon, args.train_file1)
    model_path2 = get_model_filename(args.smoother, args.lexicon, args.train_file2)

    if args.mode == TRAIN:
        log.info("Training...")
        lm = LanguageModel.make(args.smoother, args.lexicon)
        lm.set_vocab_size(args.train_file1, args.train_file2)

        lm.train(args.train_file1)
        lm.save(destination=model_path1)
        lm.train(args.train_file2)
        lm.save(destination=model_path2)
    elif args.mode == TEST:
        log.info("Testing...")
        gen_counter = 0
        total_counter = 0
        lm1 = LanguageModel.load(model_path1)
        lm2 = LanguageModel.load(model_path2)
        type1 = args.train_file1.stem.split('.')[0]
        type2 = args.train_file2.stem.split('.')[0]
        for test_file in args.test_files:
            prob_gen = lm1.file_log_prob(test_file) + math.log(args.gen_prior) - 0#??
            # print(lm1.file_log_prob(test_file))
            # print(math.log(args.gen_prior))
            prob_spam = lm2.file_log_prob(test_file) + math.log(1-args.gen_prior) - 0#??
            # print(lm2.file_log_prob(test_file))
            # print(math.log(1-args.gen_prior))
            # print("----------")
            if prob_gen >= prob_spam:
                gen_counter += 1
                print(f"{type1}\t{test_file.stem}{test_file.suffix}")
            else:
                print(f"{type2}\t{test_file.stem}{test_file.suffix}")
            total_counter += 1

        print(f"{gen_counter:g} files were more probably {type1} ({100*gen_counter/total_counter:.2f}%)")
        print(f"{total_counter-gen_counter:g} files were more probably {type2} ({100*(total_counter - gen_counter)/total_counter:.2f}%)")
    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

