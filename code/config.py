import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train a neural network for document-level event factuality prediction')
    parser.add_argument('--data_path', type=str, default='../data/chinese.xml', help='path to the data file')
    parser.add_argument('--saved_path', type=str, default='../data/chinese_uncertain_plain_gcn_joint_doc.pkl', help='path to the saved_data file')
    parser.add_argument('--n_epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='size of the training batches')
    parser.add_argument('--labmda', type=float, default=0.2)

    parser.add_argument('--gpu', dest="gpu", action="store_const", const=True, default=True, required=False, help='optional flag to use GPU if available')
    parser.add_argument('--gradient_accumulation_steps', type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--warmup_proportion', default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--gcn_layers', type=int, default=2, help="the number of gcn layers")
    parser.add_argument('--gcn_hid_dim', type=int, default=768, help='the hidden size of gcn')
    parser.add_argument('--gcn_out_dim', type=int, default=768, help='the output size of gcn')
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--linear_dim', type=int, default=300)
    parser.add_argument('--model_path', type=str, default="./checkpoint/chinese_model")
    parser.add_argument('--output_path', type=str, default="./result/chinese_output")
    parser.add_argument('--output_path1', type=str, default="./result/chinese_output_n1")
    parser.add_argument('--output_path2', type=str, default="./result/chinese_output_n2")

    args = parser.parse_args()
    for arg in vars(args):
        print('{}={}'.format(arg.upper(), getattr(args, arg)))
    print('')
    return args

opt = parse_args()
