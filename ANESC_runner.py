import argparse
import LoadData as data
from ANESC import ANESC




def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")
    parser.add_argument('--data_path', nargs='?', default='./data/washington/',
                        help='Input data path')

    parser.add_argument('--id_dim', type=int, default=20,
                        help='Dimension for id_part.')

    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Coefficient of attribute mixing.')

    parser.add_argument('--initial_gamma', type=float, default=0.001,
                        help='Initial clustering weight coefficient. .')

    parser.add_argument('--cluster_number', type=int, default=5,
                        help='Number of cluster.')

    parser.add_argument('--n_neg_samples', type=int, default=10,
                        help='Number of negative samples.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of batch_size.')

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs.')


    return parser.parse_args()

#################### Util functions ####################







def run_ANESC( data, id_dim, attr_dim,epoch,batch_size,initial_gamma,alpha,cluster_number,n_neg_samples):
    model = ANESC( data, id_embedding_size=id_dim, attr_embedding_size=attr_dim,
                 epoch=epoch,
                 batch_size=batch_size,
                 cluster_number=cluster_number,
                 alpha=alpha,
                 initial_gamma=initial_gamma,
                 n_neg_samples=n_neg_samples)
    model.train( )



if __name__ == '__main__':
    args = parse_args()
    print("data_path: ", args.data_path)
    path = args.data_path
    Data = data.LoadData( path)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)


    embedding = run_ANESC(Data, args.id_dim, args.attr_dim,
                          epoch=args.epoch,
                          cluster_number=args.cluster_number,
                          batch_size=args.batch_size,
                          initial_gamma=args.initial_gamma,
                          alpha=args.alpha,
                          n_neg_samples=args.n_neg_samples)



