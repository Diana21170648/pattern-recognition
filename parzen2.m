clear
%% 读取数据
[X1, X2, X3, X4 ,class] = textread('balancescale.txt', '%f%f%f%f%f','delimiter', ',' );
%DATA=[X1, X2, X3, X4 ,class];
%data=DATA(randperm(size(DATA, 1)),:);
dataX=[X1, X2, X3, X4];
dataY=zeros(size(class,1),size(class,2));
%% 按类别对总体样本进行计数
dataY=class;%输出的624*1
data=[dataX,dataY];%624*5
j=1;%第一列
for i=1:1:size(data,1) %size指数组大小，data代表原始数据624*5,1代表列数，%1是i的初始值，size(data,1)是终止值，1是步长。
      if dataY(i)==1
         data_1(j,:)=data(i,:);%把筛选的属于第一类的数据放进矩阵data_1，48*5
         j=j+1;
      end
end
Data_1=data_1(randperm(size(data_1, 1)),:); % 随机打乱矩阵的行数,计算结果具有随机性，更有信服力，更准确

j=1;
for i=1:1:size(data,1)
      if dataY(i)==2
         data_2(j,:)=data(i,:);%288*5
         j=j+1;
      end
end
Data_2=data_2(randperm(size(data_2, 1)),:); 

j=1;
for i=1:1:size(data,1)
      if dataY(i)==3
         data_3(j,:)=data(i,:);%288*5
         j=j+1;
      end
end
Data_3=data_3(randperm(size(data_3, 1)),:); 

%% 训练集和测试集
train1 =0.75*size(data_1,1);% 提取训练和验证数据 75%训练，% 25验证
train2 = 0.75*size(data_2,1);
train3 = 0.75*size(data_3,1);

data_E1=Data_1(1:train1,:);
data_T1=Data_1(train1+1:size(Data_1,1),:);

data_E2=Data_2(1:train2,:);
data_T2=Data_2(train2+1:size(Data_2,1),:);

data_E3=Data_3(1:train3,:);
data_T3=Data_3(train3+1:size(Data_3,1),:);

data1=[data_E1;data_E2;data_E3];%468*5，测试集的数
data2=[data_T1;data_T2;data_T3];%156*5，训练集的数

%%按类别对训练集样本进行计数，即遍历测试集数组
count=zeros(1,3);%zeros创建1行3列的全0数组
j=1;
for i=1:size(data1,1)
       if data1(i,5)==1
          count(1)=count(1)+1;
       else if data1(i,5)==2
           count(2)=count(2)+1;
           else if data1(i,5 )==3
           count(3)=count(3)+1;
               end
               j=j+1; 
           end
       end
end

%计算先验概率
for i=1:size(count,2)%size（A,1）代表返回矩阵A的列数，size（A,2）代表返回矩阵A的行数
       P(i)=count(i)/size(data1,1);%先验概率=出现的次数/总数
end
%% 非参数估计（利用25%的测试集data2）
%指数窗求概率密度函数

h=3;      %窗宽度，随机取值，越大代表越平缓函数的叠加，分辨率会降低
d=4;      %测试样本特征个数，维数（4个属性）
VN=h^d;%可取VN=1/sqrt（156） ，或VN=h/sqrt（N）
%VN=1/sqrt(156);
p1=zeros(size(data2,1),1);%156*1，1代表列数，p1，p2，p3代表类条件概率，测试集数据有156行1列
p2=zeros(size(data2,1),1);%156*1
p3=zeros(size(data2,1),1);%156*1

%   x：待估计点（data2）
%   xi:训练样本（data_EX)
for i=1:size(data2,1)
for j=1:size(data_E1,1)%求训练集E1的窗函数
u1=(data2(i,1:4)-data_E1(j,1:4))/h;
f1=0.5*exp(-abs(u1));
end
       p1(i)=1/count(1)/VN*sum(f1);%求概率密度函数
end
for i=1:size(data2,1)
for j=1:size(data_E2,1)%求训练集E2的窗函数
u2=(data2(i,1:4)-data_E2(j,1:4))/h;
f2=0.5*exp(-abs(u2));
end
       p2(i)=1/count(1)/VN*sum(f2);
end
for i=1:size(data2,1)
for j=1:size(data_E3,1)%求训练集E3的窗函数
u3=(data2(i,1:4)-data_E3(j,1:4))/h;
f3=0.5*exp(-abs(u3));
end
        p3(i)=1/count(1)/VN*sum(f3);
end
%求后验概率
P1=p1*P(1);%后验概率=（类条件概率*先验概率）/全概率密度，因为分母一样，所以只计算分子
P2=p2*P(2);
P3=p3*P(3);
%% 分类结果（分两类情况处理）
for i=1:size(data2,1)
       if P1(i)>P2(i)
          T_P(i)=P1(i);
       else T_P(i)=P2(i);
       end
       if T_P(i)<P3(i)
          T_P(i)=P3(i);
       end
end

 for i=1:size(data2,1)
       if T_P(i)==P1(i)
       T_Y(i)=1; %T_Y代表计算测试输出类别
       elseif T_P(i)==P2(i)
          T_Y(i)=2;
       elseif T_P(i)==P3(i)
          T_Y(i)=3;
       end
end
%正确计数
 count_Ture=0;
for i=1:size(data2,1)
       if T_Y(i)==data2(i,5)%测试输出类别等于所属类别，则分类正确
          count_Ture=count_Ture+1;
       end
end
fprintf('分类准确率：%f%%\n',count_Ture/size(data2,1)*100)