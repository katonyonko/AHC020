from typing import List
import io
import sys
import glob
from collections import deque
from heapq import heappop, heappush
from collections import deque
from random import randint, shuffle, uniform, choice, sample

#同じフォルダにtest_caseという名称でテストケースを置けば、test関数等が動く
path='C:/Users/katonyonko/OneDrive/デスクトップ/AHC020'
files = sorted(glob.glob(path+"/test_case/*"))

import sys
sys.setrecursionlimit(10**6)
class UnionFind():
  def __init__(self, n):
    self.n = n
    self.parents = [-1] * n
  def find(self, x):
    if self.parents[x] < 0:
      return x
    else:
      self.parents[x] = self.find(self.parents[x])
      return self.parents[x]
  def union(self, x, y):
    x = self.find(x)
    y = self.find(y)
    if x == y:
      return
    if self.parents[x] > self.parents[y]:
       x, y = y, x
    self.parents[x] += self.parents[y]
    self.parents[y] = x
  def size(self, x):
    return -self.parents[self.find(x)]
  def same(self, x, y):
    return self.find(x) == self.find(y)
  def members(self, x):
    root = self.find(x)
    return [i for i in range(self.n) if self.find(i) == root]
  def roots(self):
    return [i for i, x in enumerate(self.parents) if x < 0]
  def group_count(self):
    return len(self.roots())
  def all_group_members(self):
    return {r: self.members(r) for r in self.roots()}
  def __str__(self):
    return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

class Solver:

    def __init__(self, N: int, M: int, K: int, broadcasters: list, edges: list, residents: list):
      self.N = N
      self.M = M
      self.K = K
      self.broadcasters = broadcasters
      self.edges = edges
      self.residents = residents
      self.G=[[] for _ in range(N)]
      for i in range(M):
        u, v, w = edges[i]
        u-=1; v-=1
        self.G[u].append((w,v,i))
        self.G[v].append((w,u,i))
      self.P=[0]*N
      self.B=[0]*M
      self.Btmp=[0]*M
      self.distance=[]
      for i in range(N):
        x,y=self.broadcasters[i]
        for j in range(K):
          a,b=self.residents[j]
          self.distance.append((x-a)**2+(y-b)**2)

    def idx(self,i,j): return i*self.K+j

    def bfs(G,s,B):
      inf=10**30
      D=[inf]*len(G)
      D[s]=0
      dq=deque()
      dq.append(s)
      while dq:
        x=dq.popleft()
        for w,y,i in G[x]:
          if B[i]==0: continue
          if D[y]>D[x]+1:
            D[y]=D[x]+1
            dq.append(y)
      return D

    def Dijkstra(self,G,S,p):
      done=[False]*len(G)
      inf=10**20
      C=[inf]*len(G)
      for s in S: C[s]=0
      h=[]
      for s in S: heappush(h,(0,s))
      while h:
        x,y=heappop(h)
        if done[y]:
          continue
        done[y]=True
        for v in G[y]:
          if C[v[1]]>C[y]+v[0]:
            C[v[1]]=C[y]+v[0]
            heappush(h,(C[v[1]],v[1]))
      now=p
      while C[now]>0:
        S.append(now)
        for v in G[now]:
          if C[v[1]]<C[now]:
            self.Btmp[v[2]]=1
            now=v[1]

    def score(self, P, B):
      S=sum([P[i]**2 for i in range(self.N)])+sum([self.edges[i][2] for i in range(self.M) if B[i]==1])
      return int((10**6)*(1+(10**8)/(S+10**7)))

    def solve(self):
      for j in range(self.K):
        d=min([self.distance[self.idx(i,j)] for i in range(self.N)])
        id=[self.distance[self.idx(i,j)] for i in range(self.N)].index(d)
        self.P[id]=max(self.P[id],int((d-1)**.5)+1)
      A=[i for i in range(self.N) if self.P[i]>0 and i!=0]
      nowscore=0
      for i in range(40):
        S=[0]
        for j in range(len(A)):
          self.Dijkstra(self.G,S,A[j])
        s=self.score(self.P,self.Btmp)
        # print(*self.P)
        # print(*self.Btmp)
        # print(s)
        if s>nowscore:
          nowscore=s
          self.B=self.Btmp.copy()
        for i in range(self.M): self.Btmp[i]=0
        shuffle(A)
      print(*self.P)
      print(*self.B)

def main(i):
  if i>=0:
    with open(files[i]) as f:
      lines = f.read()
    sys.stdin = io.StringIO(lines)
  N, M, K = map(int,input().split())
  broadcasters=[tuple(map(int,input().split())) for _ in range(N)]
  edges=[tuple(map(int,input().split())) for _ in range(M)]
  residents=[tuple(map(int,input().split())) for j in range(K)]
  solver = Solver(N, M, K, broadcasters, edges, residents)
  solver.solve()

def test(s,g):
  for i in range(s,g):
    main(i)

if __name__ == "__main__":
  flg=0
  if flg==0: main(-1)
  elif flg==1: test(0,1)