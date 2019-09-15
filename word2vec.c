//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
// 代码注释解读：
// 1. https://blog.csdn.net/google19890102/article/details/51887344
// 2. https://blog.csdn.net/jingquanliang/article/details/82886645
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// 哈希，线性探测，开放定址法，装填系数0.7
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

// 定义的浮点数
typedef float real;                    // Precision of float numbers

// 词的结构体
struct vocab_word {
  long long cn; // 出现的次数
  int *point; // 从根结点到叶子节点的路径
  char *word, *code, codelen;  // 分别对应着词，Huffman编码，编码长度
};

char train_file[MAX_STRING], output_file[MAX_STRING]; // 训练文件，输出文件
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab; // 出现的词的统计


// binary 0则vectors.bin输出为二进制（默认），1则为文本形式
// cbow 1使用cbow框架，0使用skip-gram框架
// debug_mode 大于0，加载完毕后输出汇总信息，大于1，加载训练词汇的时候输出信息，训练过程中输出信息
// window 窗口大小，在cbow中表示了word vector的最大的sum范围，在skip-gram中表示了max space between words（w1,w2,p(w1 | w2)）
// min_count 设置最低频率,默认是5,如果一个词语在文档中出现的次数小于5,那么就会丢弃
// num_threads 线程数
// min_reduce ReduceVocab删除词频小于这个值的词，因为哈希表总共可以装填的词汇数是有限的
//如果词典的大小N>0.7*vocab_hash_size,则从词典中删除所有词频小于min_reduce的词。
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
// 词汇表的hash存储，下标是词的hash，内容是词在vocab中的位置，a[word_hash] = word index in vocab
int *vocab_hash;
// vocab_max_size 词汇表的最大长度，可以扩增，每次扩1000
// vocab_size 词汇表的现有长度，接近vocab_max_size的时候会扩容
// layer1_size 隐层的节点数
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
// train_words 训练的单词总数（词频累加）
// word_count_actual 已经训练完的word个数
// file_size 训练文件大小，ftell得到
// classes 输出word clusters的类别数(聚类的数目)
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
// alpha BP算法的学习速率，过程中自动调整
// starting_alpha 初始alpha值
// sample 亚采样概率的参数，亚采样的目的是以一定概率拒绝高频词，使得低频词有更多出镜率，默认为0，即不进行亚采样
//（采样的阈值，如果一个词语在训练样本中出现的频率越大,那么就越会被采样）
real alpha = 0.025, starting_alpha, sample = 1e-3;
// syn0：词向量，syn1：隐层到输出层hs的权重，syn1neg：隐层到输出层negative sampling的权重，expTable：预计算的sigmoid表
// syn0 表示： 存储词典中每个词的词向量
// syn1 表示： hs(hierarchical softmax)算法中霍夫曼编码树非叶结点的权重
// syn1neg 表示： ns(negative sampling)负采样时，存储每个词对应的辅助向量（可以参考https://blog.csdn.net/itplus/article/details/37998797）
// expTable 预先存储sigmod函数结果，算法执行中查表，提前计算好，提高效率
real *syn0, *syn1, *syn1neg, *expTable;
// start 算法运行的起始时间，会用于计算平均每秒钟处理多少词
clock_t start;
// hs 采用hs还是ns的标志位，默认采用ng
int hs = 0, negative = 5;
// table_size 静态采样表的规模
// table 采样表
const int table_size = 1e8;
int *table;

// 生成负采样的概率表
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power); //遍历词汇表，统计词的能量总值train_words_pow

  // 类似轮盘赌生成每个词的概率
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow; //表示已遍历词的能量值占总能量的比

  //a - table表的索引
  //i - 词汇表的索引
  for (a = 0; a < table_size; a++) {
    table[a] = i; //单词i占用table的a位置, table反映的是一个单词能量的分布，一个单词能量越大，所占用的table的位置越多
    if (a / (double) table_size > d1) {
      i++; //移到下个词

      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1; // 处理最后一段概率，所有落在最后一个概率区间的，都选中最后一个词
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 读取训练数据中的每一个词
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue; // 回车，\r
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) { // 当前的词还没结束
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>"); // 换行符用</s>表示
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0; // 字符串结尾添加一个0
}

// Returns hash value of a word
// 生成词的hash值
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 查找词在词库中的位置index，若没有查找到则返回-1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1; // 不存在该词
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash]; // hash的word相同,返回索引值
    hash = (hash + 1) % vocab_hash_size; // 不相同的时候线性探测
  }
  return -1; // 不存在该词
}

// Reads a word and returns its index in the vocabulary
// 返回的是在词库中的位置
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// 为词库中增加一个词: 要处理hash及冲突，词库空间不够等问题，最终返回词在词库中的索引
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1; // 单词的长度+1
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char)); //开始的位置增加指定的词
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0; // </s>没有出现过，统一设置成0，在其他地方会对词频进行+1操作
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) { // 类似C++ STL vector的思想，空间不够用了增加1000个空间
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word); // 对增加的词hash
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size; // hash的碰撞检测,线性探测-开放定址法
  vocab_hash[hash] = vocab_size - 1; // 词的hash值->词的词库中的索引
  return vocab_size - 1; // 返回当前词在词库中的索引
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 根据词出现的频率对词库中的词排序
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  // count decrease
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

  // 排完序后需要重新做hash运算
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    // 根据min_count对低频词的处理
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  // 为构建huffman树申请空间
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// 删除频率较小的词
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  // 通过min_reduce控制
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b; // 删减后词的个数

  // 重新进行hash操作
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 根据词库中的词频构建Huffman树，完成每个词从根结点到叶子节点的路径，huffman编码，编码长度
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  // leaf and inner nodes: N + L == 2 * N + 1 -> N = L - 1
  // actually, *count just need vocab_size * 2
  //count数组中前vocab_size存储的是每一个词的对应的词频，后面初始化的是很大的数，用来存储生成节点的频数
  //binary数组中前vocab_size存储的是每一个词的对应的二进制编码，后面初始化的是0，用来存储生成节点的编码
  //parent_node数组中前vocab_size存储的是每一个词的对应的父节点，后面初始化的是0，用来存储生成节点的父节点
  long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  // binary not explictly initialize to 0
  // calloc doc: Allocate NMEMB elements of SIZE bytes each, all initialized to 0.
  long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));

  // 分成两半进行初始化
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn; // 前半部分初始化为每个词出现的次数
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15; // 后半部分初始化为一个固定的常数

  // 两个指针：
  // pos1指向前半截的尾部index
  // pos2指向后半截的开始index
  pos1 = vocab_size - 1;
  pos2 = vocab_size;

  // Following algorithm constructs the Huffman tree by adding one node at a time
  // 每次增加一个节点，构建Huffman树
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    // 选择最小的节点min1, 每次寻找两个最小的点做合并，最小的点为0，词小的点为1
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }

    // 选择最小的节点min2
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }

    count[vocab_size + a] = count[min1i] + count[min2i];
    // 设置父节点
    parent_node[min1i] = vocab_size + a; //存储父节点的编号：为叶子节点数目+a，a表示当前生成第a个节点
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1; // 存储两个节点中 词频大的节点定为1，代表负类
  }
  // Now assign binary code to each vocabulary word
  // 为每一个词分配二进制编码，即Huffman编码, 顺着父子关系找回编码
  for (a = 0; a < vocab_size; a++) { // 针对每一个词
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b]; // 找到当前的节点的编码
      point[i] = b; // 记录从叶子节点到根结点的序列, 路径赋值，第一个是自己
      i++;
      b = parent_node[b]; // 找到当前节点的父节点

      // touch tree root
      // 已经找到了根结点，根节点是没有编码的
      if (b == vocab_size * 2 - 2)
        break;
    }

    // codelen is char, when vocab counts is 1, 2, 4, ... 2^n,
    // the tree's height may easy exceed 255, so change codelen to int
    // may more appropriate
    // 以下要注意的是，同样的位置，point总比code深一层
    vocab[a].codelen = i; // 编码长度赋值，少1，没有算根节点
    vocab[a].point[0] = vocab_size - 2; // 根结点

    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b]; // 编码的反转, 没有根节点，左子树0，右子树1
      vocab[a].point[i - b] = point[b] - vocab_size; // 记录的是从根结点到叶子节点的路径
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

// 读取输入的文件，并从输入文件中构建词库
void LearnVocabFromTrainFile() {
  char word[MAX_STRING]; // 存储每一个单词
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; // 初始化
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0; // 记录文件中的词的个数
  AddWordToVocab((char *) "</s>"); // 在最开始增加指定的词

  // 开始从文本取每一个词
  while (1) {
    ReadWord(word, fin); // 读取每一个词
    if (feof(fin)) break; // 判断文件是否读完
    train_words++; // 记录词的个数
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c\n", train_words / 1000, 13);
      fflush(stdout);
    }

    i = SearchVocab(word); // 查找词在词库中的位置index

    if (i == -1) { // 没有查找到对应的词
      a = AddWordToVocab(word); // 增加词
      vocab[a].cn = 1; // 设置词出现的次数为1
    } else {
      vocab[i].cn++; // 设置词出现的次数+1
    }

    // 根据当前词的个数和设定的hash表的大小，删除低频词
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab(); // 根据词出现的频率对词进行排序
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin); // 训练数据的字节数
  fclose(fin);
}

// 保存词库
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "w");
  // 保存词库时，保存的是词库中的词和词出现的次数
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; // 初始化vocab_hash
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c); // wordCount，换行
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// 初始化网络
// 主要分为两个部分：1、对词向量的初始化；2、对映射层到输出层权重的初始化
void InitNet() {
  // init matrix
  long long a, b;
  unsigned long long next_random = 1;

  // 为每一个词分配词向量的空间
  // 对齐分配内存,posix_memalign函数的用法类似于malloc的用法，最后一个参数的分配的内存的大小
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  // 层次softmax的结构
  if (hs) {
    //  Hierarchical Softmax
    // 映射层到输出层之间的权重
    a = posix_memalign((void **) &syn1, 128, (long long) vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0; // 权重初始化为0
  }

  // 负采样的结构
  if (negative > 0) {
    // negative sampling
    a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0; // 全连接的权重，初始化为0
  }

  // random init words vector
  // 随机初始化词向量
  for (a = 0; a < vocab_size; a++)
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;

      // 1、与：相当于将数控制在一定范围内
      // 2、0xFFFF：65536
      // 3、/65536：[0,1]之间
      syn0[a * layer1_size + b] =
          (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size; // 初始化词向量 [-0.5/size, 0.5/size]， size为词向量的长度
    }

  // 构建Huffman树
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1]; // word_count表示出现在词库中词的总数量
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long) id;
  real f, g;
  clock_t now;

  // layer1_size为词向量的长度
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));// 存储映射层的结果
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));// 输出层到映射层的error

  FILE *fi = fopen(train_file, "rb");
  // parallel read files
  // 利用多线程对训练文件划分，每个线程训练一部分的数据
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  // 训练模型的核心部分
  while (1) {
    if (word_count - last_word_count > 10000) { // 每处理10000个词重新计算学习率
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real) (iter * train_words + 1) * 100,
               word_count_actual / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // 重新计算alpha的值
      alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
      // 防止学习率过小
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // sentence_length=0表示的是当前还没有读取文本
    // 开始读取文本，读取词的个数最多为MAX_SENTENCE_LENGTH
    if (sentence_length == 0) {
      // 需要根据文件指针的位置读取相应的文本，读到文件结尾；sentence长度超过最大长度；读到</s>(\n)都会跳出该循环
      while (1) {
        word = ReadWordIndex(fi); // 词在词库中的索引index

        if (feof(fi)) break;
        if (word == -1) continue; // 没有查到该词
        word_count++;
        if (word == 0) break; // 表示读到的是</s>
        // The subsampling randomly discards frequent words while keeping the ranking same，高频词亚采样
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
        }
        sen[sentence_length] = word; // 存储词在词库中的位置，word代表的是Index
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break; // 达到指定长度
      }
      sentence_position = 0; // 将待处理的文本指针置0
    } // 将一个文本sentence转化成用词库中 index 表示的sentence

    // 当前的线程已经处理完分配给该线程的文本；或者说当前线程已经读完数据
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;

      // 当前线程的迭代次数
      local_iter--;
      if (local_iter == 0) break; // 迭代结束

      // 重新置0，准备下一次重新迭代
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;

      // 重置文件指针
      fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
      continue;
    }

    // sen表示的是当前的线程读取到的每一个词对应在词库中的索引
    word = sen[sentence_position]; //sentence_position表示的是当前词

    if (word == -1) continue;

    // 初始化映射层
    for (c = 0; c < layer1_size; c++) neu1[c] = 0; // 映射层的结果
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

    // 产生一个0~window-1的随机数
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window;
#ifdef FIX_SKIP_GRAM
    // junfeng added
    // see http://junfenglx.github.io/word2vec-train-procedure-explained.html#skip-gram
    if (cbow) {  //train the cbow architecture
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX

        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    else {  //train skip-gram
      // just updates w_t once
      l1 = word * layer1_size;
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;

        // HIERARCHICAL SOFTMAX

        if (hs) for (d = 0; d < vocab[last_word].codelen; d++) {
          f = 0;
          l2 = vocab[last_word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[last_word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = last_word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == last_word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
      }
      // just updates w_t once
      // Learn weights input -> hidden
      for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
    }
#else

    // 模型的训练
    if (cbow) {  // 训练CBOW模型
      // in -> hidden
      // 输入层到映射层
      // window word count
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          // sentence_position is sp
          // c in [sp - (window - b), sp + (window - b)]
          c = sentence_position - window + a; // sentence_position表示的是当前的位置

          // 判断c是否越界
          if (c < 0) continue;
          // may be is break?
          // because c is increase
          if (c >= sentence_length) continue;

          last_word = sen[c]; // 找到c对应的索引
          if (last_word == -1) continue;
          // sum window words' vectors
          for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size]; // 累加
          cw++;
        }

      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw; // 计算均值，得到映射层结果

        // 计算的中心词是word，接下来用Hierarchical Softmax或者Negative Sampling训练，默认是Negative Sampling
        // 层次Softmax
        if (hs)
          for (d = 0; d < vocab[word].codelen; d++) { // word为当前词
            // 计算输出层的输出
            f = 0;
            l2 = vocab[word].point[d] * layer1_size; // 找到第d个词对应的权重
            // l2 is index j in equation

            // Propagate hidden -> output
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2]; // 映射层到输出层

            // f is +- hs_j
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
              // approximation
              // may be round is better
            else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; // Sigmoid结果，相当于激活函数
            // 'g' is the error multiplied by the learning rate
            // why sub code[d]?
            // another sigmoid property
            // \sigma(x) = 1 - \sigma(-x)
            // code[d] is 0 or 1
            g = (1 - vocab[word].code[d] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2]; // 修改映射后的结果
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c]; // 修改映射层到输出层之间的权重
          }

        // NEGATIVE SAMPLING  负采样默认
        if (negative > 0)
          for (d = 0; d < negative + 1; d++) {
            // target is w_O, w_i
            // 标记target和label
            if (d == 0) { // 正样本
              target = word;
              label = 1;
            } else { // random sample negative weight vector 选择出负样本
              next_random = next_random * (unsigned long long) 25214903917 + 11;
              target = table[(next_random >> 16) % table_size];// 从table表中选择出负样本
              // 重新选择
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            // calculate ns
            // syn1neg[l2] is v', neu1 is v_mcw
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2]; // 映射层到输出层
            // calculate gradient
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
              // \sigma(x) = 1 - \sigma(-x), and 0 label for -ns
            else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            // neu1e is added weight vector for each w_{t+q}
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            // update v'
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
          }

        // hidden -> in
        // 以上是从映射层到输出层的修改，现在返回修改每一个词向量，即映射层到输入层的权重词向量
        for (a = b; a < window * 2 + 1 - b; a++)
          if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;

            // 利用窗口内的所有词向量的梯度之和来更新
            for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
          }
      }
    } else {  //train skip-gram 训练skip-gram模型
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          // l1 is window index t+q
          l1 = last_word * layer1_size;
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

          // HIERARCHICAL SOFTMAX
          if (hs)
            for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;
              l2 = vocab[word].point[d] * layer1_size;
              // Propagate hidden -> output
              // l1 is w_{t+q}, l2 is n(w_t, j)  映射层即为输入层
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
            }
          // NEGATIVE SAMPLING
          if (negative > 0)
            for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                // should target = last_word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long) 25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              // l1 is w_{t+q}, l2 is in [w_t, w_i \sim P_n(w)]
              // l1 should be w_t
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
              if (f > MAX_EXP) g = (label - 1) * alpha;
              else if (f < -MAX_EXP) g = (label - 0) * alpha;
              else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
    }
#endif

    // 当已经处理完读入的所有文本，要重新继续往下读文本
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

// 模型训练
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));// 多线程
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha; // 学习率指定词库，则从词库中读入词

  // 区分是否指定词库
  // 若不指定词库，则从文件中构建词库
  if (read_vocab_file[0] != 0)
    ReadVocab(); // 指定词库
  else
    LearnVocabFromTrainFile(); // 不指定词库，从文件中构建词库

  if (save_vocab_file[0] != 0) SaveVocab(); // 判断是否需要保存词库

  // 若没有指定输出文件，则退出
  if (output_file[0] == 0) return;
  // done read vocabulary

  InitNet(); // 初始化网络，完成权重初始化，huffman编码

  if (negative > 0) InitUnigramTable(); // 利用负采样的方法

  // 开始训练
  start = clock();
  for (a = 0; a < num_threads; a++) {
    pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
  }
  for (a = 0; a < num_threads; a++) {
    pthread_join(pt[a], NULL);
  }

  // output results 输出最终的训练结果
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size); // 词汇量，vector维数
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) {
        for (b = 0; b < layer1_size; b++) {
          fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        }
      } else {
        for (b = 0; b < layer1_size; b++) {
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        }
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *) malloc(classes * sizeof(int));
    int *cl = (int *) calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *) calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

// 解析命令行
int ArgPos(char *str, int argc, char **argv) {// 查找对应的参数
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a; // 匹配成功，返回值所在的位置
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  //  判断参数的个数
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf(
        "./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0; // 输出文件
  save_vocab_file[0] = 0;// 输出词的文件
  read_vocab_file[0] = 0;// 读入指定词的文件

  // 解析word2vec所需用到的参数
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word)); // 存储每一个词的结构体
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int)); // 存储词的hash
  expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real)); // 申请EXP_TABLE_SIZE+1个空间

  // excellent precompute sigmoid
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
#ifdef FIX_SKIP_GRAM
  printf("uses fixed skip-gram version\n");
#endif
  // 开始模型训练
  TrainModel();

  return 0;
}
