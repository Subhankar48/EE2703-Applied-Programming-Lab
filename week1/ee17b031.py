import sys
from os.path import exists
arguments = sys.argv

if (len(arguments)<2):
    print("\t\tToo few arguments given.Expected two.")
    sys.exit(0)
elif (len(arguments)>2):
    print("\t\tToo many arguments passed. Expected two.")
    sys.exit(0)
else:
    try:
        f = open(arguments[1], 'r')
    except Exception:
        print('''\t\tFile doesn't exist in this directory.
                Make sure that the filename is typed correctly
                and you are in the correct directory.''')
        sys.exit(0)

raw_data = f.read().splitlines()

start_index = 0;
try:
    while(raw_data[start_index]!='.circuit'): #finding the index of the '.circuit\n'
        start_index = start_index +1
except Exception:
    print("There is some issue with the file.")
    sys.exit(0)

end_index = start_index
while((raw_data[end_index]!='.end')):   #finding the index of the '.end\n'
    end_index = end_index + 1

first_refined_data = raw_data[start_index+1:end_index]  #shorten the array to keep only the required elements

final_data = []
#print(first_refined_data)
for n in first_refined_data:
    #print(n)
    final_data.append(n)

choice =input("Do you want to see the parsed values printed?\nEnter Yes then.\n")
#print(final_data)

database = []
word_array = []

def parsing_function(input_database,present_line):
    words = present_line.split(' ')
    

    words = list(filter(lambda a: a!='', words))        #for some reason i was getting junk spaces after using split. this is to remove those spaces.
    #print(words)
    for p in range(len(words)):
        if (words[p][0] == '#'):
            words = words[0:p]
            break
    #print(words)
    word_array.append(words)
    temp_dictionary = { }   #create an empty dictionary
    N = len(words)
    if (N==4):
        
        temp_dictionary['Type'] = 'Independent'
        temp_dictionary['Name'] = words[0][0:2]
        if (words[0][0] == 'R'):
            temp_dictionary["Element"] = 'Resistor'
        
        elif (words[0][0] == 'L'):
            temp_dictionary["Element"] = 'Inductor'
        
        elif (words[0][0] == 'C'):
            temp_dictionary["Element"] = 'Capacitor'
        
        elif (words[0][0] == 'V'):
            temp_dictionary["Element"] = 'Independent Voltage Source'

        elif (words[0][0] == 'I'):
            temp_dictionary["Element"] = "Independent Current Source"

        else:
            temp_dictionary["Element"] = "Unknown Element"
        
        temp_dictionary['From Node'] = words[1]
        temp_dictionary['To Node'] = words[2]
        temp_dictionary["Value"] = float(words[3])

    if (N==6):
        temp_dictionary["Type"] = "Dependent"
        temp_dictionary['Name'] = words[0][0:2]
        if (words[0][0] == 'E'):
            temp_dictionary["Element"] = "VCVS"
        
        elif (words[0][0] == 'G'):
            temp_dictionary["Element"] = "VCCS"

        elif (words[0][0] == 'H'):
            temp_dictionary["Element"] = "CCVS"
        
        elif (words[0][0] == 'F'):
            temp_dictionary["Element"] = "CCCS"
        
        else:
            temp_dictionary["Element"] = "Unknown Element"
        
        temp_dictionary['From Node'] = words[1]
        temp_dictionary['To Node'] = words[2]
        temp_dictionary['Dependent From Node'] = words[3]
        temp_dictionary['Dependent To Node'] = words[4]
        temp_dictionary["Value"] = float(words[5])

    input_database.append(temp_dictionary)
    if ((choice =='Yes')|(choice == 'yes')|(choice=='YES')):
    	print(f"{temp_dictionary}\n")

for a in final_data:
    parsing_function(database,a)
print(word_array)

if ((choice =='Yes')|(choice == 'yes')|(choice=='YES')):
    print("Legend\n")
    print("VCCS : Voltage Controlled Current Source\n")
    print("VCVS : Voltage Controlled Voltage Source\n")
    print("CCVS : Current Controlled Voltage Source\n")
    print("CCCS : Current Controlled Current Source\n")

choice2 = input("Do you want to see the file values printed in reverse order?\nEnter Yes then.\n")
if ((choice2 =='Yes')|(choice2 == 'yes')|(choice2=='YES')):
    reverse_list = list(reversed(word_array))
    for element in reverse_list:
        reverse_words = list(reversed(element))
        temp_string = ''
        for word in reverse_words:
            temp_string = temp_string+' '+word
        print(f"{temp_string}")