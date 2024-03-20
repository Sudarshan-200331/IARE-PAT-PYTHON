#task scheduler
def leastinterval(tasks,n):
    taskcount=[0]*26
    for task in tasks:
        taskcount[ord(task)-ord('A')]+=1
    taskcount.sort(reverse=True)
    maxcount=taskcount[0]-1
    idleslots=maxcount*n
    for i in range(1,len(taskcount)):
        idleslots-=min(taskcount[i],maxcount)
    idleslots=max(0,idleslots)
    return len(tasks)+idleslots
#____________________________________________________________________________

# maximum length of pair chain
def findlongestchain(pairs):
    pairs.sort(key=lamba x:x[1])
    print(pairs)
    currentend=float('-inf')
    longestchain=0
    for pair in pairs:
        if pair[0]>currentend:
            currentend=pair[1]
            longestchain+=1
    return longestchain
#____________________________________________________________________________

#split array into consecutive subsequences
def ispossible(nums):
    count={}
    for num in nums:
        count[num]=count.get(num,0)+1
    end={}
    for num in nums:
        if count[num]==0:
            continue
        count[num]-=1
        if end.get(num-1,0)>0:
            end[num-1]-=1
            end[num]=end.get(num,0)+1
        elif count.get(num+1,0)>0 and count.get(num+2,0)>0:
            count[num+1]-=1
            count[num+2]-=1
            end[num+2]=end.get(num+2,0)+1
        else: return False
    return True
#____________________________________________________________________________

#maximum swap
def maximumswap(num):
    listnum=[x for x in str(num)]
    print(listnum)
    p1,p2=0,0
    maxdigit=len(listnum)-1
    for i in range(len(listnum)-1,-1,-1):
        if listnum[i]>listnum[maxdigit]:
            maxdigit=i
        elif listnum[i]<listnum[maxdigit]:
            p1,p2=i,maxdigit
    listnum[p1],listnum[p2]=listnum[p2],listnum[p1]
    return int(" ".join(listnum))
#____________________________________________________________________________

#valid parenthesis string
def checkvalidstring(s):
    lo=hi=0
    for c in s:
        lo+=1 if c=='(' else -1
        hi+=1 if c!=')' else -1
        if hi<0:break
        l0=max(lo,0)
    return lo==0
#____________________________________________________________________________

#monotone increasing digits
def monotoneincreasingdigits(n):
    listnum=list(str(n))
    flag=len(listnum)
    for i in range(len(listnum)-1,0,-1):
        if listnum[i-1]>listnum[i]:
            flag=i
            listnum[i-1]=str(int(listnum[i-1])-1)
    for i in range(flag,len(listnum)):
        listnum[i]='9'
    return int(" ".join(listnum))
#____________________________________________________________________________

#rabbit in forest
from collections import defaultdict
import math
def numrabbits(answers):
    unique=defaultdict(int)
    for a in answers:
        unique[a]+=1
    tot=0
    for sim,count in unique.items():
        basenum=sim+1
        tot
#____________________________________________________________________________

#minimum no of platflorm required
def findplatform(arr,dep,n):
    arr.sort()
    dep.sort()
    platneeded=1
    result=1
    i=1
    j=0
    while(i<n and j<n):
        if (arr[i]<=dep[i]):
            platneeded+=1
            i+=1
        elif (arr[i]>dep[i]):
            platneeded-=1
            j+=1
        if (platneeded>result):
            result=platneeded
    return result
#____________________________________________________________________________

#job sequencing
def printjobsequencing(arr,t):
    n=len(arr)
    for i in range(n):
        for j in range(n-1-i):
            if (arr[j][2]<arr[j+1][2]):
                arr[j],arr[j+1]=arr[j+1],arr[j]
    result=[False]*t
    job=['-1']*t
    for i in range(len(arr)):
        for j in range(min(t-1,arr[i][1]-1),-1,-1):
            if (result[j] is False):
                result[j]
#____________________________________________________________________________

#fibonacci series using recursion
def fibonacci(n):
    if n<=1:
        return n
    return fibonacci(n-1)+fibonacci(n-2)
#____________________________________________________________________________

#fibonacci series using dynamic programming
def fibonacci(n):
    f=[0,1]
    for i in range(2,n+1):
        f.append(f[i-1]+f[i-2])
    return f[n]
#____________________________________________________________________________

#fibonacci series using space optimisation
def fibonacci(n):
    a=0
    b=1
    if n<0:
        print('incorrect output')
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        for i in range(2,n+1):
            
#_________________________________________________________________________

#maximum subarray
def maxsubarray(nums):
    maxsub=nums[0]
    currentsum=0
    for n in nums:
        if currentsum<0:
            currentsum=0
        currentsum+=n
        maxsub=max(maxsub,currentsum)
    return maxsub 
#____________________________________________________________________________

#
def maxwater(height):
    ans,i,j=0,0,len(height)-1
    while (i<j):
        if height[i]<=height[j]:
            res=height[i]*(j-i)
            i+=1
        else:
            res=height[i]*(j-i)
            j-=1
        if res>ans:ans=res
    return ans
#____________________________________________________________________________

#longest common prefix
def longestcommonprefix(strs):
    l=list(zip(*strs))
    prefix=''
    for i in l:
        if len(set(i))==1:
            prefix+=i[0]
        else:
            break
    return prefix
#____________________________________________________________________________

#
def maxprofit(prices):
    if not prices:return 0
    maxprofit=0
    minpurchase=prices[0]
    for i in range(1,len(prices)):
        maxprofit=max(maxprofit,prices[i]-minpurchase)
        minpurchase=min(minpurchase,prices[i])
    return maxprofit
#____________________________________________________________________________

#best time to buy & sell stock II
def maxprofit(prices):
    profit=0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            profit+=(prices[i]-prices[i-1])
    return profit
#____________________________________________________________________________

#best time to buy & sell stock III
def maxprofit(prices):
    initial=0
    bought1=float('-inf')
    sold1=0
    bought2=float('-inf')
    sold2=0
    for x in prices:
        bought1=max(initial-x,bought1)
        sold1=max(bought+x,sold1)
        bought2=max(sold1-x,bought2)
        sold2=max(bought2+x,sold2)
    return sold2
#____________________________________________________________________________

#best time to buy & sell stock with transaction fee
def maxprofit(prices,fee):
    pos=-prices[0]
    profit=0
    n=len(prices)
    for i in range(1,n):
        pos=max(pos,profit-prices[i])
        profit=max(profit,pos+prices[i]-fee)
    return profit
#____________________________________________________________________________

#longest palindromic substring
def longestpalindrome(s):
    longestpalindrome=' '
    dp=[[0]*len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i]=True
        longestpalindrome=s[i]
    for i in range(len(s)-1,-1,-1):
        for j in range(i+1,len(s)):
            if s[i]==s[j]:
                if j-i==1 or dp[i+1][j-1] is True:
                    dp[i][j]=True
                    if len(longestpalindrome)<len(s[i:j+1]):
                        longestpalindrome=s[i:j+1]
    return longestpalindrome
#____________________________________________________________________________

#valid palindrome
def ispalindrome(s):
    b=s.lower()
    l=0
    r=len(b)-1
    while l<r:
        if not b[l].isalnum():l+=1
        if not b[r].isalnum():r-=1
        if b[r].isalnum() and b[l].isalnum():
            if b[l]!=
#____________________________________________________________________________

#rotate the matrix 90 degress clockwise direction
def rotate(matrix):
    n=len(matrix[0])
    for i in range(n//2+n%2):
        for j in range(n//2):
            tmp=matrix[n-1-j][i]
            matrix[n-1-j][i]=matrix[n-1-i][n-j-1]
            matrix[n-1-i][n-j-1]=matrix[j][n-1-i]
            matrix[j][n-1-i]=matrix[i][j]
            matrix[i][j]=tmp
def showmatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            print(matrix[i][j],end=" ")
        print()
#____________________________________________________________________________

#rotate the matrix 90 degrees anticlockwise
def rotatematrix(mat,N):
    for x in range(0,int(N/2)):
        for y in range(x,N-x-1):
            temp=mat[x][y]
            mat[x][y]=mat[y][N-1-x]
            mat[y][N-1-x]=mat[N-1-x][N-1-y]
            mat[N-1-x][N-1-y]=mat[N-1-y][x]
            mat[N-1-y][x]=temp

#____________________________________________________________________________

#
def rotatematrix(mat):
    if not len(mat):
        return
    top=0
    bottom=len(mat)-1
    left=0
    right=len(mat[0])-1
    while left<right and top<bottom:
        prev=mat[top+1][left]
        for i in range(left,right+1):
            curr=mat[top][i]
            mat[top][i]=prev
            prev=curr
        top+=1
        for i in range(top,bottom+1):
            curr=mat[i][right]
            mat[i][right]=prev
            prev=curr
        right-=1
        for i in range(right,left-1,-1):
            curr=mat[bottom][i]
            mat[bottom][i]=prev
            prev=curr
        bottom-=1
        for i in range(bottom,top-1,-1):
            curr=mat[i][left]
            mat[i][left]=prev
            prev=curr
        left+=1
    return mat
#____________________________________________________________________________

#spiral order
def spiralorder(matrix):
    if len(matrix)==0:
        return matrix
    left=0
    right=len(matrix[0])-1
    top=0
    down=len(matrix)-1
    ans=[]
    while top<=down and left<=right:
        for i in range(left,right+1):
            ans.append(matrix[top][i])
        for i in range(top+1,down+1):
            ans.append(matrix[i][right])
        for i in reversed(range(left,right)):
            if top==down:
                break
            ans.append(matrix[down][i])
        for i in reversed(range(top+1,down)):
            if left==right:
                break
            ans.append(matrix[i][left])
        top+=1
        down-=1
        left+=1
        right-=1
    return ans
#____________________________________________________________________________

#diagonal sum
def diagonalsum(mat):
    size=len(mat)
    sum=0
    for i in range(size):
        sum+=mat[i][i]+mat[i][size-i-1]
    if size%2!=0:
        sum-=mat[size//2][size//2]
    return sum
#____________________________________________________________________________

#integer to roman
def intoroman(num):
    m=["","M","MM","MMM"]
    c=["","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"]
    x=["","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"]
    i=["","I","II","III","IV","V","VI","VII","VIII","IX"]
    thousands=m[num//1000]
    hundreds=c[(num%1000)//100]
    tens=x[(num%100)//10]
    ones=i[num%10]
    ans=(thousands+hundreds+tens+ones)
    ans=ans.replace(" "," ")
    return ans
#____________________________________________________________________________

#roman to integer
roman={"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
def romantoint(s):
    sum=0
    for i in range(len(s)-1,-1,-1):
        num=roman[s[i]]
        if 3*num<sum:
            sum=sum-num
        else:
            sum=sum+num
    return sum
#____________________________________________________________________________

#sort characters by freq
import heapq
def freqsort(s):
    freq={}
    for char in s:
        freq[char]=freq.get(char,0)+1
    maxheap=[(-freq[char],char) for char in freq]
    sortedstring=''
    while maxheap:
        count,char=heapq.heappop(maxheap)
        sortedstring+=char*(-count)
    return sortedstring
#____________________________________________________________________________

#swim in raising water
import heapq
def swiminwater(grid):
    n=len(grid)
    visit=set()
    minh=[[grid[0][0],0,0]]
    directions=[[0,1],[0,-1],[1,0],[-1,0]]
    visit.add((0,0))
    while minh:
        time,r,c=heapq.heappop(minh)
        if r==n-1 and c==n-1:return time
        for dr,dc in directions:
            neir,neic=r+dr,c+dc
        if (neir<0 or neic<0 or neir==n or neic==n or (neir,neic) in visit:continue
        visit.add((neir,neic))
        heapq.heappush(minh,[max(time,grid[neir][neic],neir,neic)]
        print(minh)
#____________________________________________________________________________
                       
#car pooling
import heapq
def carpooling(trips,capacity):
    heap=[]
    for x,y,z in trips:
        heap.extend([(y,x),(z,-x)])
    heapq.heapify(heap)
    while capacity>=0 and heap:
        capacity-=heapq.heappop(heap)[1]
    return len(heap)==0
#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________
                       
#

#____________________________________________________________________________

                    
