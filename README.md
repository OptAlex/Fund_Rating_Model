# What are we doing?
Perform substantial market risk modelling according to SVB and Lehman 

# How to use GIT?

1. Never push to main branch! This branch is only used for merging new features into the main code. The features are tested on the prep branch first!
2. We only merge the prep branch into the main branch with at least two people checking the code and functionality.
3. The prep branch is the prototyping branch where we test new features and how they integrate in the currently used code. 
4. If you want to create a new feature (a new functionality of the code) you do the following steps:
  4.1 Create a branch called "feat/Description of what the purpose of the branch is" from main
  4.2 This is the branch where you work on the new feature. Make sure to change your branch in the IDE of your choice.
  4.3 Each commit to the branch has to follow the commit rules provided below.
5. Merge your feature into the prep branch. 

# Committing/Pushing:
1. There are rules on how to structure your push messages:
  - feat: A new feature
  - fix: A bug fix
  - docs: Documentation only changes
  - style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  - refactor: A code change that neither fixes a bug nor adds a feature
  - perf: A code change that improves performance
  - test: Adding missing tests
  - chore: Changes to the build process or auxiliary tools and libraries such as documentation generation
2. Formatting: we use black formatting for the code (https://pypi.org/project/black/) before committing
3. Make sure to get rid of unnecessary comments. Nobody needs a commit such as "looping over all risk factors. We all can read a loop.
4. Speaking Loops: get rid of useless loops. You can use ChatGPT or Copilot to get better performing code.
5. After your code is tested successfully and being pushed to the main, add your NEW packages to the requierments.txt file. Make sure to specify your version in cases needed

# Example
You want to add a feature which calculates the option price using Black-Scholes
1. Create a branch "feat/BlackScholesOptionPricing" from main
2. Change the branch in your IDE to the new branch
3. Implement your Code.
  3.1 Make sure to use existing help functions and to create a new BlackScholesOptionPricing.py file.
  3.2 Commit, using the rules provided, after certain points (f.e. after long call, after long put, etc.)
5. Check your code for comments and unnecessary for loops, etc. 
6. Use black formatting 
7. Push your commits to your branch: "feat/BlackScholesOptionpricing, extended description (field): add BS Option Pricing functionality"
  NOTE: discription is always simple present!
9. check code and results with someone else and merge your branch into prep
10. modify requirements.py
11. If everything is tested and OK, merge prep into main (keep the prep branch!)
