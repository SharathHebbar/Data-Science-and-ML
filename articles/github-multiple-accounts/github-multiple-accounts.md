# Access Git(Github/Gitlab .. ) using multiple accounts on the same system.
Often you need to use multiple GitHub/GitLab accounts from the same system and you use the command line to do this. But while trying this you will face so many errors as you can only add one user to use git services, so below are the steps.

Step 1:
Install the Git command-line tool on your system, if it is not already installed.
The link to download git is as follows, you can choose your package according to your OS
https://git-scm.com/download

Step 2:
Generate an SSH key for each of the GitHub accounts you want to use. You can do this by running the following command, replacing “email@example.com” with the email address associated with your GitHub account:
Note: You can see .ssh folder in C:\Users\[Your Username]\.ssh 
if not exists then it will automatically create .ssh folder
ssh-keygen -t rsa -b 4096 -C “email@example.com”
Step 3:
When prompted, enter a file in which to save the key. It is recommended to use a different file for each key, so you can easily distinguish between them. For example, you might use "id_rsa_account1" for one account and "id_rsa_account2" for another.
Ex: C:\Users\[Your Username]\.ssh \id_rsa -> C:\Users\[Your Username]\.ssh \id_rsa_account1
Step 4:
Follow the prompts to enter a passphrase for the key. This passphrase will be used to encrypt the private key, so make sure to choose a strong and unique passphrase.
You can even not give any passphrase
Two files will be generated for each key
Id_rsa_account1
id_rsa_account1.pub

Step 5:
Once the key has been generated, copy the public key (.pub) and paste it in your github account ssh section.
[In Github go to settings > SSH and GPG > New SSH key > Add Title of your choice > In key section paste the public key ]

Step 6:
Repeat the steps above for each of the GitHub accounts you want to use. 
Step 7: 
Create a config file in the same directory (.ssh)

<!---
account 1
--> 
Host github.com-personal
   HostName github.com
   User git
   IdentityFile ~/.ssh/id_rsa_account1

<!---
account 2
-->
Host github.com-work
   HostName github.com
   User git
   IdentityFile ~/.ssh/id_rsa_account2
Step 8:
Now Initialize a git repository in any of the desired location
Then add all the required files that needs to be committed
Then commit those files with appropriate commit message
Then while adding remote repo Take ssh url and change github.com to github.com-account1
Ex: git remote add origin git@github.com-account1:dummy-proj/dummy-proj.git
You can do the same for git clone too



 



