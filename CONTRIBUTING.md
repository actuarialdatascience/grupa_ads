## Git

### Installation
https://www.linode.com/docs/development/version-control/how-to-install-git-on-linux-mac-and-windows/

### Clone repository
```
git clone https://github.com/dawidkopczyk/grupa_ads.git
```

### Contribute
1. Do changes

2. Commit changes (make sure you are in the right folder grupa_ads)
```
git add .
git commit -m some_name
```

3. Create local branch
```
git checkout -b branch_name
```

4. Push from local branch to remote branch
```
git push origin branch_name
```

5. Go to Github and create pull request. 

6. Come back to master and delete branch
```
git checkout master
git branch -d branch_name
```