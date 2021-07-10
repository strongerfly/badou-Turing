print('hello world!')
 # token
#  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

# commit push  ggyysophia<1040311149@qq.com>   第xx次作业 commit
# push
# 看是否提交成功
# 1. git log 看最近一次提交
# 2. 看自己的github仓库
# 发起pullRequst
# 1. 在自己的githu仓库中， 找到 Pull requests
# 2. 确定自己的仓库和老师的仓库， 一定是自己的指向老师的
# 3. 写自己的一些作业新的， 或者向老师请教的问题，点击create pull request
# 4. 看到老师的仓库显示了你的最新的Pull request, 即视为提交成功

# 三、后续仓库的更新问题
# 大家每次提交作业的时候， 老师的Github QAboard的仓库都会更新， 每次更新的时候，
# 我们都要将自己的fork的远程代码进行更新
# 用git命令即可：
# 1. 输入命令 git remote -v
# 查看远程的分支情况， 如果你的upstream不如图片中的2两一样
# 先输入命令 git remote remove upstream
# 2. 再添加老师侧的远程仓库
# 不需要：git remote add upstream https://github.com/ggyysophia/badou-Turing.git
# 3. 更新老师侧的代码， 即从源仓库更新同步代码， 并合并到本地
#  git fetch upstreamgit pull origin master
#  git merge upstream/master
# 4. 更新并合并自己远程仓库的代码
# git pull origin master
# 5. 向自己的远程推送刚才同步源仓库后的代码
# git push







