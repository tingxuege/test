[无重复字符的最大子串](#1)
[最长公共前缀](#2)
[第一个字符串的排列之一是第二个字符串的子串](#3)
[字符串相乘](#4)
[翻转字符串里的单词](#5)
[简化路径](#6)
[复原ip地址](#7)
[三数之和](#8)
[岛屿的最大面积](#9)
[搜索旋转排序数组 - 照升序排序的数组在预先未知的某个点上进行了旋转](#10)
[最长连续递增序列](#11)
[数组中的第K个最大元素](#12)
[最长连续序列](#13)
[第K个排列](#14)
[朋友圈](#15)
[合并区间](#16)
[接雨水](#17)
[合并两个有序链表](#18)
[反转链表](#19)
[链表两数相加](#20)
[排序链表](#21)
[环形链表II](#22)
[相交链表](#23)
[合并K个排序链表](#24)
[二叉树的最近公共祖先](#25)
[二叉树的锯齿层次遍历](#26)
[股票-最多只允许完成一笔交易（即买入和卖出一支股票一次）](#27)
[股票-多次买卖一支股票](#28)
[最大正方形-找到只包含 1 的最大正方形，并返回其面积](#29)
[最大子序和-找到一个具有最大和的连续子数组](#30)
[三角形最小路径和](#31)
[俄罗斯套娃信封](#32)










<h5 id="1">无重复字符的最长子串</h5>
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        int res = 0;
        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0, j = 0;j < len;j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            res = Math.max(res, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return res;
    }
}
```

<h5 id="2">最长公共前缀</h5>
```java
public String longestCommonPrefix(String[] strs) {
   if (strs.length == 0) return "";
   String prefix = strs[0];
   for (int i = 1; i < strs.length; i++)
       while (strs[i].indexOf(prefix) != 0) {
           prefix = prefix.substring(0, prefix.length() - 1);
           if (prefix.isEmpty()) return "";
       }        
   return prefix;
}
```

```java
public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";

        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }
        });

        String str = strs[0];
        StringBuffer s = new StringBuffer();
        OUT:
        for (int i = 0;i < str.length();i++) {
           for(int j = 1;j < strs.length;j++) {
               if(strs[j].charAt(i) != str.charAt(i)) break OUT;
           }
           s.append(str.charAt(i));
        }
        return s.toString();
    }
```

<h5 id="3">第一个字符串的排列之一是第二个字符串的子串</h5>
```java
class Solution {
//我们可以使用更简单的数组数据结构来存储频率，而不是仅使用特殊的哈希表数据结构来存储字符出现的频率。给定字符串仅包含小写字母（'a'到'z'）。因此我们需要采用大小为 26 的数组。其余过程与最后一种方法保持一致。
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        int[] s1map = new int[26];
        for (int i = 0; i < s1.length(); i++)
            s1map[s1.charAt(i) - 'a']++;
        for (int i = 0; i <= s2.length() - s1.length(); i++) {
            int[] s2map = new int[26];
            for (int j = 0; j < s1.length(); j++) {
                s2map[s2.charAt(i + j) - 'a']++;
            }
            if (matches(s1map, s2map))
                return true;
        }
        return false;
    }
    public boolean matches(int[] s1map, int[] s2map) {
        for (int i = 0; i < 26; i++) {
            if (s1map[i] != s2map[i])
                return false;
        }
        return true;
    }
}
```
```java
//我们可以为s2 中的第一个窗口创建一次哈希表，而不是为s2中考虑的每个窗口重新生成哈希表。然后，稍后当我们滑动窗口时，我们知道我们添加了一个前面的字符并将新的后续字符添加到所考虑的新窗口中。因此，我们可以通过仅更新与这两个字符相关联的索引来更新哈希表。同样，对于每个更新的哈希表，我们将哈希表的所有元素进行比较以获得所需的结果。
public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        int[] s1map = new int[26];
        int[] s2map = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            s1map[s1.charAt(i) - 'a']++;
            s2map[s2.charAt(i) - 'a']++;
        }
        for (int i = 0; i < s2.length() - s1.length(); i++) {
            if (matches(s1map, s2map))
                return true;
            s2map[s2.charAt(i + s1.length()) - 'a']++;
            s2map[s2.charAt(i) - 'a']--;
        }
        return matches(s1map, s2map);
    }
    public boolean matches(int[] s1map, int[] s2map) {
        for (int i = 0; i < 26; i++) {
            if (s1map[i] != s2map[i])
                return false;
        }
        return true;
    }
```
```java
//上一种方法可以优化，如果不是比较每个更新的 s2maps2map 的哈希表的所有元素，而是对应于 s2s2 考虑的每个窗口，我们会跟踪先前哈希表中已经匹配的元素数量当我们向右移动窗口时，只更新匹配元素的数量。
//为此，我们维护一个 countcount 变量，该变量存储字符数（26个字母表中的数字），这些字符在 s1s1 中具有相同的出现频率，当前窗口在 s2s2 中。当我们滑动窗口时，如果扣除最后一个元素并添加新元素导致任何字符的新频率匹配，我们将 countcount 递增1.如果不是，我们保持 countcount 完整。但是，如果添加频率相同的字符（添加和删除之前）相同的字符，现在会导致频率不匹配，这会通过递减相同的 countcount 变量来考虑。如果在移动窗口后，countcount 的计算结果为26，则表示所有字符的频率完全匹配。所以，我们立即返回一个True。
public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        int[] s1map = new int[26];
        int[] s2map = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            s1map[s1.charAt(i) - 'a']++;
            s2map[s2.charAt(i) - 'a']++;
        }
        int count = 0;
        for (int i = 0; i < 26; i++)
            if (s1map[i] == s2map[i])
                count++;
        for (int i = 0; i < s2.length() - s1.length(); i++) {
            int r = s2.charAt(i + s1.length()) - 'a', l = s2.charAt(i) - 'a';
            if (count == 26)
                return true;
            s2map[r]++;
            if (s2map[r] == s1map[r])
                count++;
            else if (s2map[r] == s1map[r] + 1)
                count--;
            s2map[l]--;
            if (s2map[l] == s1map[l])
                count++;
            else if (s2map[l] == s1map[l] - 1)
                count--;
        }
        return count == 26;
    }
```


<h5 id="4">字符串相乘</h5>
```java
public String multiply(String num1, String num2) {
        BigInteger a = new BigInteger(num1);
        BigInteger b = new BigInteger(num2);
        a = a.multiply(b);
        
        return a.toString();
    }
```
```java
class Solution {
    /**
    * 计算形式
    *    num1
    *  x num2
    *  ------
    *  result
    */
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        // 保存计算结果
        String res = "0";
        
        // num2 逐位与 num1 相乘
        for (int i = num2.length() - 1; i >= 0; i--) {
            int carry = 0;
            // 保存 num2 第i位数字与 num1 相乘的结果
            StringBuilder temp = new StringBuilder();
            // 补 0 
            for (int j = 0; j < num2.length() - 1 - i; j++) {
                temp.append(0);
            }
            int n2 = num2.charAt(i) - '0';
            
            // num2 的第 i 位数字 n2 与 num1 相乘
            for (int j = num1.length() - 1; j >= 0 || carry != 0; j--) {
                int n1 = j < 0 ? 0 : num1.charAt(j) - '0';
                int product = (n1 * n2 + carry) % 10;
                temp.append(product);
                carry = (n1 * n2 + carry) / 10;
            }
            // 将当前结果与新计算的结果求和作为新的结果
            res = addStrings(res, temp.reverse().toString());
        }
        return res;
    }

    /**
     * 对两个字符串数字进行相加，返回字符串形式的和
     */
    public String addStrings(String num1, String num2) {
        StringBuilder builder = new StringBuilder();
        int carry = 0;
        for (int i = num1.length() - 1, j = num2.length() - 1;
             i >= 0 || j >= 0 || carry != 0;
             i--, j--) {
            int x = i < 0 ? 0 : num1.charAt(i) - '0';
            int y = j < 0 ? 0 : num2.charAt(j) - '0';
            int sum = (x + y + carry) % 10;
            builder.append(sum);
            carry = (x + y + carry) / 10;
        }
        return builder.reverse().toString();
    }
}
```

<h5 id="5">翻转字符串的单词</h5>
```java
class Solution {
    public String reverseWords(String s) {
        String[] str = s.trim().split(" ");
        StringBuffer newString = new StringBuffer();
        for(int i = str.length - 1;i >= 0;i--) {
            if(str[i].equals("")) continue;
            newString.append(str[i]);
            if(i != 0) newString.append(" ");
        }
        return newString.toString();
    }
}
```
```java
class Solution {
    public String reverseWords(String s) {
        int left = 0, right = s.length() - 1;
        // 去掉字符串开头的空白字符
        while (left <= right && s.charAt(left) == ' ') ++left;

        // 去掉字符串末尾的空白字符
        while (left <= right && s.charAt(right) == ' ') --right;

        Deque<String> d = new ArrayDeque();
        StringBuilder word = new StringBuilder();
        
        while (left <= right) {
            char c = s.charAt(left);
            if ((word.length() != 0) && (c == ' ')) {
                // 将单词 push 到队列的头部
                d.offerFirst(word.toString());
                word.setLength(0);
            } else if (c != ' ') {
                word.append(c);
            }
            ++left;
        }
        d.offerFirst(word.toString());

        return String.join(" ", d);
    }
}
```

<h5 id="6">简化路径</h5>
```java
class Solution {
    public String simplifyPath(String path) {
        String[] paths = path.split("/");
        StringBuilder str = new StringBuilder();

        Stack<String> stack = new Stack<>();

        for (String p : paths) {
            if (p.equals(".")) continue;
            if (p.equals("..")) {
                if (!stack.isEmpty()) stack.pop();
            } else if (!p.equals("")) stack.push(p);
        }
        if (stack.isEmpty()) return "/";
        else {
            for (int i = 0;i < stack.size();i++) str.append("/" + stack.get(i));
        }
        return str.toString();
    }
}
```
<h5 id="7">复原ip地址</h5>
```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        String[] tmp = new String[4];
        if (s.length() < 4 || s.length() > 12) return res;

        getIpAddress(s, 0, res, tmp, 0);
        return res;
    }

    public void getIpAddress(String s, int index, List<String> res, String[] tmp, int point) {
        if (point == 3) {
            if (index < s.length() && judgeIP(s, index, s.length())) {
                tmp[3] = s.substring(index, s.length());
                String str = tmp[0] + "." + tmp[1] + "." + tmp[2] + "." + tmp[3];
                res.add(str);
            }
            return;
        } else {
            for(int i = 0;i < 3;i++) {
                if(judgeIP(s, index, index + i + 1)) {
                    //System.out.println(index + " " + i + " " + point);
                    tmp[point] = s.substring(index, index + i + 1);
                    getIpAddress(s, index + i + 1, res, tmp, point + 1);
                }
            }
        }
        return;
    }

    public boolean judgeIP(String s, int index, int end) {
        if(end > s.length()) return false;
        s = s.substring(index, end);
        if((s.charAt(0) == '0' && s.length() > 1) || Integer.parseInt(s) > 255) return false;
        return true;
    }
}
```
<h5 id="8">三数之和</h5>
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        if(nums.length < 3) return res;
        Arrays.sort(nums);
        int i = 0, j = 0;
        int sum = 0;

        for(int index = 0;index < nums.length;index++) {
            if(nums[index] > 0) return res;
            if(index > 0 && nums[index - 1] == nums[index]) continue;
            i = index + 1;
            j = nums.length - 1;

            while(i < j) {
                sum = nums[i] + nums[j] + nums[index];
                if(sum == 0) {
                    res.add(Arrays.asList(nums[i],nums[j],nums[index]));
                    while (i < j && nums[i] == nums[i + 1]) i++;
                    while (i < j && nums[j] == nums[j - 1]) j--;
                    i++;
                    j--;
                }
                else if (sum < 0) i++;
                else j--;
            }
        }
        return res;
    }
}
```

算法流程：
1.特判，对于数组长度 n，如果数组为null 或者数组长度小于 3，返回 [][]。
2.对数组进行排序。
3.遍历排序后数组：
	若 nums[i]>0：因为已经排序好，所以后面不可能有三个数加和等于 0，直接返回结果。
	对于重复元素：跳过，避免出现重复解
	令左指针L=i+1，右指针 R=n−1，当 L<R 时，执行循环：
		当 nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将 L,R 移到下一位置，寻找新的解
		若和大于 0，说明 nums[R]太大，R 左移
		若和小于 0，说明 nums[L] 太小，L 右移
空间复杂度：O(1)

<h5 id="9">岛屿的最大面积</h5>
```java
class Solution {
    public int val = 0;

    public int maxAreaOfIsland(int[][] grid) {
        if(grid.length == 0) return 0;

        for(int i = 0;i < grid.length;i++) {
            for(int j = 0;j < grid[0].length;j++) {
                if(grid[i][j] == 1) {
                    val = Math.max(val, dfs(grid, i, j, 0));
                }

            }
        }

        return val;
    }

    public int dfs(int[][] grid, int x, int y, int res) {
        if(x == grid.length || x < 0) {
            return 0;
        }
        if(y == grid[0].length || y < 0) {
            return 0;
        }
        if(grid[x][y] == 1) {
            grid[x][y] = 0;
            return 1 + dfs(grid, x + 1, y, res + 1) + dfs(grid, x, y + 1, res + 1) + 
            dfs(grid, x - 1, y, res + 1) + dfs(grid, x, y - 1, res + 1);
        }

        return 0;
    }
}
```

<h5 id="10">搜索旋转排序数组</h5>
```java
class Solution {
    public int search(int[] nums, int target) {
        int mid = 0;
        int left = 0;
        int right = nums.length - 1;
        while(left <= right) {
        	mid = left + (right - left) / 2;
        	if(nums[mid] == target) return mid;
        	else if(nums[mid] < nums[right]) {
        		if(nums[mid] <= target && target <= nums[right]) left = mid + 1;
        		else right = mid - 1;
        	}  else {
        		if(nums[mid] >= target && target >= nums[left]) right = mid - 1;
        		else left = mid + 1;
        	}
        }
        return -1;
    }
}
```
题目要求算法时间复杂度必须是 O(\log n)O(logn) 的级别，这提示我们可以使用二分搜索的方法。

但是数组本身不是有序的，进行旋转后只保证了数组的局部是有序的，这还能进行二分搜索吗？答案是可以的。

可以发现的是，我们将数组从中间分开成左右两部分的时候，一定有一部分的数组是有序的。拿示例来看，我们从 6 这个位置分开以后数组变成了 [4, 5, 6] 和 [7, 0, 1, 2] 两个部分，其中左边 [4, 5, 6] 这个部分的数组是有序的，其他也是如此。

这启示我们可以在常规二分搜索的时候查看当前 mid 为分割位置分割出来的两个部分 [l, mid] 和 [mid + 1, r] 哪个部分是有序的，并根据有序的那个部分确定我们该如何改变二分搜索的上下界，因为我们能够根据有序的那部分判断出 target 在不在这个部分：

如果 [l, mid - 1] 是有序数组，且 target 的大小满足[nums[l],nums[mid])，则我们应该将搜索范围缩小至 [l, mid - 1]，否则在 [mid + 1, r] 中寻找。
如果 [mid, r] 是有序数组，且 target 的大小满足(nums[mid+1],nums[r]]，则我们应该将搜索范围缩小至 [mid + 1, r]，否则在 [l, mid - 1] 中寻找。

<h5 id="11">最长连续递增序列</h5>
```java
class Solution {
//滑动窗口
    public int findLengthOfLCIS(int[] nums) {
        int ans = 0, anchor = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (i > 0 && nums[i-1] >= nums[i]) anchor = i;
            ans = Math.max(ans, i - anchor + 1);
        }
        return ans;
    }
}
```

<h5 id="12">数组中的第K个大元素</h5>
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        // init heap 'the smallest element first'
        PriorityQueue<Integer> heap =
            new PriorityQueue<Integer>((n1, n2) -> n1 - n2);

        // keep k largest elements in the heap
        for (int n: nums) {
          heap.add(n);
          if (heap.size() > k)
            heap.poll();
        }

        // output
        return heap.poll();        
  }
}
```

<h5 id="13">最长连续序列</h5>
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;

        int count = 1, res = 1;

        Arrays.sort(nums);
        for(int i = 1;i < nums.length;i++) {
            if(nums[i] == nums[i - 1]) continue;
            if(nums[i] - nums[i - 1] == 1) count++;
            else {
                res = Math.max(res, count);
                count = 1;
            }
        }
        return Math.max(res, count);
    }
}
```
```java
class Solution {
//这个优化算法与暴力算法仅有两处不同：这些数字用一个 HashSet 保存（或者用 Python 里的 Set），实现 O(1)O(1) 时间的查询，同时，我们只对 当前数字 - 1 不在哈希表里的数字，作为连续序列的第一个数字去找对应的最长序列，这是因为其他数字一定已经出现在了某个序列里。
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }

        int longestStreak = 0;

        for (int num : num_set) {
            if (!num_set.contains(num-1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.contains(currentNum+1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }
}
```

<h5 id="14">第K个排列</h5>
```java
class Solution {
//解题思路：将 n! 种排列分为：n 组，每组有 (n - 1)!个排列，根据k值可以确定是第几组的第几个排列，选取该排列的第1个数字，然后递归从剩余的数字里面选取下一个数字，直到n=1为止。
    public String getPermutation(int n, int k) {
        boolean[] visited = new boolean[n];        
        // 将 n! 种排列分为：n 组，每组有 (n - 1)! 种排列
        return recursive(n, factorial(n - 1), k, visited);
    }

    /**
    * @param n 剩余的数字个数，递减
    * @param f 每组的排列个数
    */
    private String recursive(int n, int f, int k, boolean[]visited){
        int offset = k%f;// 组内偏移量
        int groupIndex = k/f + (offset > 0 ? 1 : 0);// 第几组
        // 在没有被访问的数字里找第 groupIndex 个数字
        int i = 0;
        for(; i < visited.length && groupIndex > 0; i++){
            if(!visited[i]){
                groupIndex--;
            }
        }
        visited[i-1] = true;// 标记为已访问
        if(n - 1 > 0){
            // offset = 0 时，则取第 i 组的第 f 个排列，否则取第 i 组的第 offset 个排列
            return String.valueOf(i) + recursive(n-1, f/(n - 1), offset == 0 ? f : offset, visited);
        }else{
            // 最后一数字
            return String.valueOf(i);
        }
    }

    /**
    * 求 n!
    */
    private int factorial(int n){
        int res = 1;
        for(int i = n; i > 1; i--){
            res *= i;
        }
        return res;
    }

}
```

<h5 id="15">朋友圈</h5>
```java
class Solution {
    public int findCircleNum(int[][] M) {
        int len = M.length;
        int count = 0;
        int[] visited = new int[len];
        Arrays.fill(visited, 0);
        
        for(int i = 0;i < len;i++) {
            if(visited[i] == 0) {
                dfs(visited, M, i);
                count++;
            }
        }
        return count;
    }
    
    public void dfs(int[] visited, int[][] M, int index) {
        for(int j = 0;j < M.length;j++) {
            if(M[index][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                dfs(visited, M, j);
            }
        }
    }
}
```
<h5 id="16">合并区间</h5>
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if(intervals.length == 0) return intervals;

        List<int[]> list = new ArrayList<>();
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o1[1] - o2[1] : o1[0] - o2[0];
            }
        });

        list.add(intervals[0]);
        for(int i = 1;i < intervals.length;i++) {
            if(intervals[i][0] > list.get(list.size() - 1)[1]) {
                list.add(intervals[i]);
            } else {
                list.get(list.size() - 1)[1] = Math.max(intervals[i][1], list.get(list.size() - 1)[1]);
            }
        }
        return list.toArray(new int[0][]);

    }
}
```

<h5 id="17">接雨水</h5>
```java
class Solution {
    public int trap(int[] height) {
        if (height.length == 0) return 0;

        Stack<Integer> stack = new Stack<>();
        int count = 0;

        for(int i = 0;i < height.length;i++) {
        // 如果栈顶元素一直相等，那么全都pop出去，只留第一个。
            while(!stack.isEmpty() && height[stack.peek()] < height[i]) {
                int cur = height[stack.pop()];
                while (!stack.isEmpty() && height[stack.peek()] == cur) {
                    stack.pop();
                }
                if(!stack.isEmpty()) {
                // stackTop此时指向的是此次接住的雨水的左边界的位置。右边界是当前的柱体，即i。
                // Math.min(height[stackTop], height[i]) 是左右柱子高度的min，减去height[curIdx]就是雨水的高度。
                // i - stackTop - 1 是雨水的宽度。
                    count += (Math.min(height[i], height[stack.peek()]) - cur) * (i - stack.peek() - 1);
                }
            }
            stack.push(i);
        }
        return count;
    }
}
```

<h5 id="18">合并两个有序链表</h5>
```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(0);
        ListNode p = res;
        while(l1 != null && l2 != null) {
            if(l1.val > l2.val) {
                p.next = l2;
                l2 = l2.next;
            } else {
                p.next = l1;
                l1 = l1.next;
            }
            p = p.next;
        }
        p.next = l1 == null ? l2 : l1;
        return res.next;
    }
}
```
```java
//递归
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        else if (l2 == null) {
            return l1;
        }
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }

    }
}
```

<h5 id="19">反转链表</h5>
```java
//在遍历列表时，将当前节点的 next 指针改为指向前一个元素。由于节点没有引用其上一个节点，因此必须事先存储其前一个元素。在更改引用之前，还需要另一个指针来存储下一个节点。不要忘记在最后返回新的头引用！
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}
```
```java
//递归版本稍微复杂一些，其关键在于反向工作。假设列表的其余部分已经被反转，现在我该如何反转它前面的部分？

假设列表为：n1 - nk-1 - nk - nk+1 ... nm
若从节点nk+1到nm已经被反转，而我们正处于nk
n1 - nk-1 -> nk -> nk+1 <- ... <- nm
我们希望nk+1的下一个节点指向nk 
所以nk.next.next=nk
n1的下一个必须指向 Ø 。如果你忽略了这一点，你的链表中可能会产生循环。如果使用大小为 2 的链表测试代码，则可能会捕获此错误。
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode p = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return p;
}
```

<h5 id="20">两数相加链表</h5>
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode list = new ListNode(0);
        ListNode p = list;
        int carrier = 0;
        int val = 0;
        
        while(l1 != null || l2 != null) {
            val = (l1 != null ? l1.val : 0) + (l2 != null ? l2.val : 0) + carrier;
            carrier = val / 10;
            val %= 10;
            p.next = new ListNode(val);
            p = p.next;
            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
        }
        if(carrier != 0) p.next = new ListNode(carrier);
        return list.next;
    }
    
}
```

<h5 id="21">最长公共前缀</h5>
本题要求在 O(n log n) 时间复杂度和常数级空间复杂度下排序
	意味着不能用递归方法求解，那么只有迭代法
	还有O(n log n)的时间复杂度表示只能用归并排序（Merge Sort）
接下来梳理一下算法思路：
	首先要先确立链表的长度，然后通过for循环对链表元素进行倍速遍历（每次将上一步的两部分合为一部分）
	定义first和second指针指向每一部分的头结点，每次事先保存每一部分的后继结点再将其断链送入Merge进行排序
	将排序好的链表用pre指针连接起来，再将剩下的remain部分链表挂到尾部进入下一轮翻倍循环
	等到最后一轮for循环结束，如有剩余元素就挂在已排序链表尾部，如果恰好等分排好整条链表，就返回dummy.next即可。
```java
class Solution {
    public ListNode sortList(ListNode head) {
        // 确定链表长度
        int len = 0;
        ListNode cur = head;
        while (cur != null) {
            len++;
            cur = cur.next;
        }
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        
        // 每次结合的元素翻倍实现由零到整的过程
        for (int m = 1; m < len; m *= 2) {
            ListNode pre = dummy;
            cur = pre.next;
            // 每次取出两部分归并排序后合并成一个部分
            while (cur != null) {
                // 定义 first 指针指向第一部分
                ListNode first = cur;
                for (int i = 0; i < m - 1 && cur != null; i++) {
                    cur = cur.next;
                }
                if (cur == null) {
                    break;
                }
                // 定义 second 指针指向第二部分
                ListNode second = cur.next;
                // 将第一部分和第二部分断开
                cur.next = null;
                cur = second;
                for (int i = 0; i < m - 1 && cur != null; i++) {
                    cur = cur.next;
                }
                ListNode remain = null;
                // 将第二部分与第三部分断开
                if (cur != null) {
                    remain = cur.next;
                    cur.next = null;
                }
                cur = remain;
                // 准备开始归并排序,并用 tmp 指向归并后的头结点
                ListNode tmp = merge(first, second);
                pre.next = tmp;
                // pre 结点遍历到归并链表的末尾
                while (pre.next != null) {
                    pre = pre.next;
                }
                // 接上剩下未排序的元素
                pre.next = remain;
            }
        }
        return dummy.next;
    }
    
    private ListNode merge(ListNode a, ListNode b) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        while (a != null && b != null ) {
            if (a.val < b.val) {
                cur.next = a;
                cur = cur.next;
                a = a.next;
            } else {
                cur.next = b;
                cur = cur.next;
                b = b.next;
            }
        }
        if (a != null) {
            cur.next = a;
        }
        if (b != null) {
            cur.next = b;
        }
        return pre.next;
    }
}
```

<h5 id="22">环形链表</h5>
```java
public class Solution {
//首先，我们分配一个 Set 去保存所有的列表节点。我们逐一遍历列表，检查当前节点是否出现过，如果节点已经出现过，那么一定形成了环且它是环的入口。否则如果有其他点是环的入口，我们应该先访问到其他节点而不是这个节点。其他情况，没有成环则直接返回 null 。
//算法会在遍历有限个节点后终止，这是因为输入列表会被分成两类：成环的和不成环的。一个不成欢的列表在遍历完所有节点后会到达 null - 即链表的最后一个元素后停止。一个成环列表可以想象成是一个不成环列表将最后一个 null 元素换成环的入口。
//如果 while 循环终止，我们返回 null 因为我们已经将所有的节点遍历了一遍且没有遇到重复的节点，这种情况下，列表是不成环的。对于循环列表， while 循环永远不会停止，但在某个节点上， if 条件会被满足并导致函数的退出。
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> visited = new HashSet<ListNode>();

        ListNode node = head;
        while (node != null) {
            if (visited.contains(node)) {
                return node;
            }
            visited.add(node);
            node = node.next;
        }

        return null;
    }
}
```

<h5 id="23">相交链表</h5>
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        
        ListNode p = headA;
        ListNode q = headB;
        
        while(p != q) {
            if(q.next == p.next) return p.next;
            p = p.next == null ? headB : p.next;
            q = q.next == null ? headA : q.next;
        
        }
        return p;
    }
}
```

<h5 id="24">合并k个排序链表</h5>
```java
class Solution {
//优先队列
   public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                if (o1.val < o2.val) return -1;
                else if (o1.val == o2.val) return 0;
                else return 1;
            }
        });
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        for (ListNode node : lists) {
            if (node != null) queue.add(node);
        }
        while (!queue.isEmpty()) {
            p.next = queue.poll();
            p = p.next;
            if (p.next != null) queue.add(p.next);
        }
        return dummy.next;
    }
}
```
```java
class Solution {
   public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int left, int right) {
        if (left == right) return lists[left];
        int mid = left + (right - left) / 2;
        ListNode l1 = merge(lists, left, mid);
        ListNode l2 = merge(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1,l2.next);
            return l2;
        }
    }
}
```

<h5 id="25">二叉树的最近公共祖先</h5>
```java
class Solution {
    TreeNode res = null;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs(root, p, q);
        return res;
    }

    public boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return false;

        boolean left = dfs(root.left, p, q);
        boolean right = dfs(root.right, p, q);
        boolean mid = (root == p || root == q) ? true : false;

        if(mid ? (left || right) : (left && right)) res = root;
        
        return mid || left || right;
    }
}
```

<h5 id="26"> 二叉树的锯齿形层次遍历</h5>
```java
class Solution {
    List<List<Integer>> list = new ArrayList<>();
    
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if(root == null) return list;
        
        getResult(root, 0);
        for(int i = 0;i < list.size();i++) {
            if(i % 2 != 0) Collections.reverse(list.get(i));
        }
        return list;
    }
    
    public void getResult(TreeNode root, int level) {
        if(root == null) return;
        
        if(level == list.size()) {
            list.add(new ArrayList<Integer>());
        }
        
        list.get(level).add(root.val);
        
        if(root.left != null) getResult(root.left, level + 1);
        if(root.right != null) getResult(root.right, level + 1);
        
        return;
    }
}
```

<h5 id="26"> 二叉树的锯齿形层次遍历</h5>
```java

```
