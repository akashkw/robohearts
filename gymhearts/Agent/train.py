from .models import model_factory, save_model
import torch


def train(args, data):
    model = model_factory[args.model]()

    """
    data is expected to be a list of tuples, (ft_vec, return)

    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.L1Losss()

    train_data = data
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        for ft_vec, score in train_data:
            ft_vec, score = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, score)

            loss_vals.append(loss_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        print('epoch %-3d \t loss = %0.3f '% (epoch, avg_loss))
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['mlp'], default='mlp')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)
