import sys
from os.path import exists
import os
arguments = sys.argv
import numpy as np


if (len(arguments)<2):
    print("\t\tToo few arguments given.Expected two.")          #Checking the number of command line arguments
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
                and you are in the correct directory.''')           #Ensuring that the file exists
        print(os.system('ls'))
        print('''\t\tThese are the files in the present directory
                Make sure you entered a correct filename.''')
        sys.exit(0)

raw_data = f.read().splitlines()

start_index = 0
try:
    while(raw_data[start_index]!='.circuit'): #finding the index of the '.circuit'
        start_index = start_index +1
except Exception:
    print("The .circuit word is missing. Check the netlist file.")     #if we don't find the .circuit word, there is some issue with the file
    sys.exit(0)

end_index = start_index

try:
    while((raw_data[end_index]!='.end')):   #finding the index of the '.end'
        end_index = end_index + 1
except Exception:
    print("The .end word is missing. Check the netlist file.")
    sys.exit(0)
final_data= raw_data[start_index+1:end_index]  #shorten the array to keep only the required elements


ac_info_draft = raw_data[end_index+1:]
ac_info=[]
for variable in ac_info_draft:
    temporary = variable.split(' ')
    temporary= list(filter(lambda p: p!='', temporary))
    for q in range(len(temporary)):
        if temporary[q][0] =='#':
            temporary = temporary[0:q]
            break
    ac_info.append(temporary)

if ((len(ac_info)>0)&(any(len(hee)>0 for hee in ac_info))):
    for v in range(len(ac_info)):
        if (len(ac_info[v])>0):
            if(ac_info[v][0] == '.ac'):
                exists = 1
                needed_info = ac_info[v]
    if exists ==1:
        for var in needed_info:
            if var.isdigit()== True:                #Checking if it is an AC Source
                frequency = float(var)              #If not, set the frequency to a very small value
        print("The Frequency is:\n")
        print(frequency)
        print('\n')
        w = 2*np.pi*frequency
else:
    w = 0.00000000000000001


database = []
word_array = []

def parsing_function(input_database,present_line):
    words = present_line.split(' ')
    
    words = list(filter(lambda a: a!='', words))        #for some reason i was getting junk spaces after using split. this is to remove those spaces.
    for p in range(len(words)):
        if (words[p][0] == '#'):
            words = words[0:p]
            break

    word_array.append(words)
    temp_dictionary = { }   #create an empty dictionary
    N = len(words)
    if (N==4):
        
        temp_dictionary['Type'] = 'Impedance'
        temp_dictionary['Name'] = words[0][0:3]
        if (words[0][0] == 'R'):
            temp_dictionary["Element"] = 'Resistor'
        
        elif (words[0][0] == 'L'):
            temp_dictionary["Element"] = 'Inductor'
        
        elif (words[0][0] == 'C'):
            temp_dictionary["Element"] = 'Capacitor'
        
        else:
            temp_dictionary["Element"] = "Unknown Element"
        
        temp_dictionary['From Node'] = words[1]
        temp_dictionary['To Node'] = words[2]
        temp_dictionary["Value"] = float(words[3])

    elif (N==5):
        temp_dictionary["Type"] = "DC Source"
        temp_dictionary["Name"] = words[0][0:3]
        if (words[0][0] == 'V'):
            temp_dictionary["Element"] = "Independent Voltage Source"
        
        elif (words[0][0] == 'I'):
            temp_dictionary["Element"] = "Independent Current Source"

        else:
            temp_dictionary["Element"] = "Unknown Element"
        temp_dictionary["From Node"] =words[1]
        temp_dictionary["To Node"] = words[2]
        temp_dictionary["AC or DC"] =words[3]
        temp_dictionary["Value"] = float(words[4])

    elif (N==6):
        temp_dictionary["Type"] = "AC Source"
        temp_dictionary['Name'] = words[0][0:3]
        if (words[0][0] == 'V'):
            temp_dictionary["Element"] = "Independent Voltage Source"
        
        elif (words[0][0] == 'I'):
            temp_dictionary["Element"] = "Independent Current Source"

        else:
            temp_dictionary["Element"] = "Unknown Element"
        temp_dictionary['From Node'] = words[1]
        temp_dictionary['To Node'] = words[2]
        temp_dictionary['AC or DC'] = words[3]
        temp_dictionary['Value'] = float(words[4])/2
        temp_dictionary["Phase"] = float(words[5])
        phase = temp_dictionary["Phase"]
        net = temp_dictionary["Value"]*(np.cos(phase)+1j*np.sin(phase))
        temp_dictionary["Value"] = net

    else:
        print("Unknown element encountered.")
        sys.exit(0)
        
    input_database.append(temp_dictionary)
for a in final_data:
    parsing_function(database,a)

#this is the end of the data parsing and storing part of the code.
#below we build the incidence matrix

nodes = set([])
for element in database:
    nodes.add(element['From Node'])
    nodes.add(element['To Node'])

node_dict = {}
count= 1
node_dict[0] = 'GND'
for node in nodes:
    if node!='GND':
        node_dict[count] = node
        count = count+1

independent_voltage_sources =set([])
for element in database:
    if element["Element"] == "Independent Voltage Source":
        independent_voltage_sources.add(element["Name"])

source_dict = {}
for source in independent_voltage_sources:
    source_dict[count] = source
    count = count +1

for element2 in database:
        if element2["Element"] == 'Capacitor':
            element2["Value"] = -1j/(w*element2['Value'])
        if element2["Element"] == 'Inductor':
            element2["Value"] = 1j*w*element2["Value"]

incidence_matrix = np.zeros(count*count, dtype = np.complex).reshape(count,count)
y = np.zeros(count, dtype= np.complex).reshape(count,1)
#w = float(input("Enter the value of Omega.\n"))
print("Legend\n")
print("Nodes and their Indices\n")
print(node_dict)
print('\n')
print("Voltage sources and their Indices\n")
print(source_dict)
print('\n')
def equality(m,n):
    return(1 if m==n else -1)
K = len(source_dict)
#Building up the incidence matrix for Impedances
N = len(node_dict)
for n in range(1,N):
    temp = []
    for something in database:
        if ((something['From Node'] ==node_dict[n])|(something["To Node"]==node_dict[n])):
            temp.append(something)

    sum1 =0.0+0.0j
    for element in temp:
        if (element["Element"] == "Resistor")|(element["Element"] == 'Inductor')|(element["Element"] == 'Capacitor'):
            sum1 = sum1 +1/element["Value"]
    incidence_matrix[n,n]+=sum1

    for element in temp:
        if element["Element"] == 'Independent Current Source':
            y[n]=y[n] -equality(element["From Node"], node_dict[n])*element["Value"]
        temp_node=(element['To Node'] if element['From Node'] == node_dict[n]  else element["From Node"])
        for p in range(N):
            if node_dict[p] == temp_node:
                idx = p
                if ((element["Element"]!='Independent Voltage Source')&(element["Element"]!="Independent Current Source")):                    
                    incidence_matrix[n,idx] += 1/(element["Value"])*equality(n,idx)
#For the Independent Voltage Sources
for m in range(N,N+K):
    temp1 = []
    for something1 in database:
        if (something1["Name"] == source_dict[m]):
            temp1.append(something1)
    for source in temp1:
        for g in range(N):
            if source['From Node'] == node_dict[g]:
                from_idx = g
            if source["To Node"] == node_dict[g]:
                to_idx = g
        if (node_dict[from_idx]!='GND'):
            incidence_matrix[m, from_idx] =1
        if (node_dict[to_idx]!='GND'):
            incidence_matrix[m,to_idx]=-1
        y[m] = source["Value"]
        incidence_matrix[from_idx,m] = 1
        incidence_matrix[to_idx,m] = -1

y[0] = 0
incidence_matrix[0] = 0
incidence_matrix[0,0]=1
x=np.linalg.solve(incidence_matrix,y)

print('Incidence Matrix\n')
print(f"{incidence_matrix}\n")
print('Y')
print(f"{y}\n")
print("Answers")
print(f"{x}\n")
for n in range(N):
    print("The voltage at node",node_dict[n],"is",str(x[n]),"V.\n" )

for k in range(K):
    print("The current though source", source_dict[N+k] ,"is.",str(x[N+k]),'A.\n')