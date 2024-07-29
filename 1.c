def is_vowel(character):

    vowels = 'aeiouAEIOU'
    return character in vowels

def count_vowels_and_consonants(input_string):
    vowel_count = 0
    consonant_count = 0

    for char in input_string:
        if char.isalpha():
            if is_vowel(char):
vowel_count += 1
            else:
                consonant_count += 1
return vowel_count, consonant_count

def main():
    user_input = input("Enter a string: ")
    vowels, consonants = count_vowels_and_consonants(user_input)
    print(f"Number of vowels: {vowels}")
    print(f"Number of consonants: {consonants}")

if __name__ == "__main__":
    main()
