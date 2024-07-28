Code for the paper [A Deep Generative Approach to Native Language Identification](https://aclanthology.org/2020.coling-main.159/)

##Main Arguments:

- data : expects the train and test data as .csv files in `--train_data_path` and `--test_data_path`. For the cross-validation setting (cv), the `test_data_path` can be empty.
- eval_setting : can be one of `cv` (for cross-validation) or `train-test` (for the standard train-test setting). In the former `num_folds` determines the number of folds.
- lr : learning rate needs to be set higher than what commonly used for fine-tuning (~ 1.5e-4).

##Citation:

> @inproceedings{lotfi-etal-2020-deep,
> title = "A Deep Generative Approach to Native Language Identification",
> author = "Lotfi, Ehsan  and
> Markov, Ilia  and
> Daelemans, Walter",
> editor = "Scott, Donia  and
> Bel, Nuria  and
> Zong, Chengqing",
> booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
> month = dec,
> year = "2020",
> address = "Barcelona, Spain (Online)",
> publisher = "International Committee on Computational Linguistics",
> url = "https://aclanthology.org/2020.coling-main.159",
> doi = "10.18653/v1/2020.coling-main.159",
> pages = "1778--1783",
> abstract = "Native language identification (NLI) {--} identifying the native language (L1) of a person based on his/her writing in the second language (L2) {--} is useful for a variety of purposes, including marketing, security, and educational applications. From a traditional machine learning perspective,NLI is usually framed as a multi-class classification task, where numerous designed features are combined in order to achieve the state-of-the-art results. We introduce a deep generative language modelling (LM) approach to NLI, which consists in fine-tuning a GPT-2 model separately on texts written by the authors with the same L1, and assigning a label to an unseen text based on the minimum LM loss with respect to one of these fine-tuned GPT-2 models. Our method outperforms traditional machine learning approaches and currently achieves the best results on the benchmark NLI datasets.",
> }
