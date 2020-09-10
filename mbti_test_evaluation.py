
# Prepare test set

energy_test_labels = [key.get(l) for l in energy_test['energy'].tolist()]
information_test_labels = [key.get(l) for l in information_test['information'].tolist()]
organisation_test_labels = [key.get(l) for l in organisation_test['organisation'].tolist()]
decision_test_labels = [key.get(l) for l in decision_test['decision'].tolist()]


def format_test_sets(test_set, labels, name="", batch_size = 32):

    """
    formats the test sets into dataloaders for myers-briggs classifcation. 
    this function is just a helper for displaying the number of points per class. 
    uses the format_dataset function in the 'xlnet_encoding.py' file. 
    params: 
    - test_set: dataframe of test emails
    - list: list of labels (binary)
    - name: name of MB dimension being classified
    - batch size: default 32 as reliable size for usage limits.
    useful for large MBTI test sets when we want to know how many instances in each test case. 
    """

    print("Encoding Test Set..\n")

    input_ids, lengths, attention_masks = format_dataset(test_set, labels)

    if name=="Energy":
        print('Test labels: {:>10,} Extroversion (1)'.format(np.sum(labels)))
        print('Test labels: {:>10,} Introversion (0)'.format(len(labels ) -np.sum(labels)))
    elif name=="Information":
        print('Test labels: {:>10,} Intuition (1)'.format(np.sum(labels)))
        print('Test labels: {:>10,} Sensing (0)'.format(len(labels ) -np.sum(labels)))
    elif name=="Organisation":
        print('Test labels: {:>10,} Judging (1)'.format(np.sum(labels)))
        print('Test labels: {:>10,} Percieving (0)'.format(len(labels ) -np.sum(labels)))
    elif name=="Decision":
        print('Test labels: {:>10,} Thinking (1)'.format(np.sum(labels)))
        print('Test labels: {:>10,} Feeling (0)'.format(len(labels ) -np.sum(labels)))
    else:
        print('Test labels: {:>10,} positive'.format(np.sum(labels)))
        print('Test labels: {:>10,} negative'.format(len(labels ) -np.sum(labels)))
    print()

    labels = torch.tensor(labels)
    input_tensor = torch.tensor(input_ids)
    length_tensor = torch.tensor(lengths)
    attention_mask_tensor = torch.tensor(attention_masks)

    test_data = TensorDataset(input_tensor, attention_mask_tensor, labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_data, test_sampler, test_dataloader


