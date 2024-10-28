import random
import matplotlib.pyplot as plt
from ID3 import ID3, prune, test  # Make sure these are correctly implemented
import parse  # Make sure this module exists and works correctly

def testPruningOnHouseData(inFile):
    withPruning_avg = []
    withoutPruning_avg = []
    train_sizes = list(range(10, 301, 10))
    data = parse.parse(inFile)
    
    for train_size in train_sizes:
        print(f"Training size: {train_size}")
        withPruning = []
        withoutPruning = []
        
        for _ in range(100):
            random.shuffle(data)
            train = data[:train_size]
            valid_size = min(len(data)//4, len(data) - train_size)
            valid = data[train_size:train_size + valid_size]
            test_data = data[train_size + valid_size:]
            
            tree = ID3(train, "")  # Removed 'democrat' argument, adjust if necessary
            acc_test = test(tree, test_data)
            
            # Prune the tree using validation data
            pruned_tree = prune(tree, valid)
            acc_pruned_test = test(pruned_tree, test_data)
            
            withPruning.append(acc_pruned_test)
            withoutPruning.append(acc_test)
        
        avg_with_pruning = sum(withPruning) / len(withPruning)
        avg_without_pruning = sum(withoutPruning) / len(withoutPruning)
        withPruning_avg.append(avg_with_pruning)
        withoutPruning_avg.append(avg_without_pruning)
    
    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, withPruning_avg, label='With Pruning', marker='o', linestyle='-', alpha=0.7)
    plt.plot(train_sizes, withoutPruning_avg, label='Without Pruning', marker='x', linestyle='-', alpha=0.7)
    plt.title('Average Test Accuracy vs Training Set Size')
    plt.xlabel('Training Set Size')
    plt.ylabel('Average Test Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits for full range of accuracy
    plt.grid(True)
    plt.legend()
    plt.savefig("pruning_vs_no_pruning_accuracy_avg_training_size.png")
    plt.show()
    
    print("Average Accuracy with Pruning across different training sizes:", withPruning_avg)
    print("Average Accuracy without Pruning across different training sizes:", withoutPruning_avg)

def testPruningCars():
    withPruning_avg = []
    withoutPruning_avg = []
    train_sizes = list(range(10, 301, 10))

    train_cars = parse.parse("cars_train.data")
    valid_cars = parse.parse("cars_valid.data")
    test_cars = parse.parse("cars_test.data")

    print()
    print()

    tree = ID3(train_cars, "")  # Removed 'democrat' argument, adjust if necessary
    acc_test = test(tree, test_cars)

    print(f"Accuracy on Cars Test Dataset Before Pruning: {acc_test}")
            
    # Prune the tree using validation data
    pruned_tree = prune(tree, valid_cars)
    acc_pruned_test = test(pruned_tree, test_cars)

    print(f"Accuracy on Cars Test Dataset After Pruning: {acc_pruned_test}")

    print()
    print()
            
    # withPruning.append(acc_pruned_test)
    # withoutPruning.append(acc_test)
    
    # # Plot the learning curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_sizes, withPruning_avg, label='With Pruning', marker='o', linestyle='-', alpha=0.7)
    # plt.plot(train_sizes, withoutPruning_avg, label='Without Pruning', marker='x', linestyle='-', alpha=0.7)
    # plt.title('Average Test Accuracy vs Training Set Size')
    # plt.xlabel('Training Set Size')
    # plt.ylabel('Average Test Accuracy')
    # plt.ylim(0, 1)  # Set y-axis limits for full range of accuracy
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("pruning_vs_no_pruning_accuracy_avg_training_size.png")
    # plt.show()
    
    # print("Average Accuracy with Pruning across different training sizes:", withPruning_avg)
    # print("Average Accuracy without Pruning across different training sizes:", withoutPruning_avg)

# Run the experiment
# testPruningOnHouseData("house_votes_84.data")
testPruningCars()