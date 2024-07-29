def input_list():
    result = []
    index = 0
    while True:
        user_input = input(f'Enter element {index + 1} (or type "exit" to finish): ')
        if user_input.lower() == "exit":
            break
        try:
            element = int(user_input)
            result.append(element)
            index += 1
        except ValueError:
            print("Please enter a valid integer or 'exit' to finish.")
    return result

def find_common_elements(list_a, list_b):
    common_elements = set(list_a) & set(list_b)
    return common_elements

def count_common_elements(list_a, list_b):
    common_elements = find_common_elements(list_a, list_b)
    return len(common_elements)

def main():
    print('Enter elements for the first list:')
    list_a = input_list()

    print('Enter elements for the second list:')
    list_b = input_list()

    common_count = count_common_elements(list_a, list_b)
    print(f'Number of common elements: {common_count}')

if __name__ == "__main__":
    main()
