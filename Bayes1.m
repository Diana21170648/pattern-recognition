clc
clear
%% 读取数据
[x1, x2, x3, x4,class] = textread('balancescale.txt', '%f%f%f%f%f','delimiter', ',');
dataX=[x1, x2, x3, x4];
dataY=zeros(size(class,1),size(class,2)); %1返回行数，2返回列数。624*1的零矩阵
%% 按类别对总体样本进行计数（看每类分别有多少组）
dataY=class;
data=[dataX,dataY]; %624*5
j=1;
for i=1:size(data,1)
      if dataY(i)==1
         data_1(j,:)=data(i,:);%把原数据集里筛选的第一类放进矩阵data_1，有48组
         j=j+1;
      end
end
j=1;
for i=1:size(data,1)
      if dataY(i)==2
         data_2(j,:)=data(i,:);%把筛选的第二类放进矩阵data_2，有288组
         j=j+1;
      end
end
j=1;
for i=1:size(data,1)
      if dataY(i)==3
         data_3(j,:)=data(i,:);%把筛选的第三类放进矩阵data_3，有288组
         j=j+1;
      end
end
%% 训练集和测试集
%每类取前75%作为测试集，总计468个样本放入data1
nbertrain1 = 0.75*size(data_1,1);               % 提取训练和验证数据 75%训练，25% 验证
nbertrain2 = 0.75*size(data_2,1);
nbertrain3 = 0.75*size(data_3,1);
Data_1=data_1(randperm(size(data_1, 1)),:);     % 随机打乱矩阵的行数
data_E1=Data_1(1:nbertrain1,:);                 %前75%作为训练集
data_T1=Data_1(nbertrain1+1:size(Data_1,1),:);  %后25%作为测试集
Data_2=data_2(randperm(size(data_2, 1)),:); 
data_E2=Data_2(1:nbertrain2,:);
data_T2=Data_2(nbertrain2+1:size(Data_2,1),:);
Data_3=data_3(randperm(size(data_3, 1)),:);
data_E3=Data_3(1:nbertrain3,:);
data_T3=Data_3(nbertrain3+1:size(Data_3,1),:);
data1=[data_E1;data_E2;data_E3];     %训练集
data2=[data_T1;data_T2;data_T3];     %测试集
%% bayes参数估计
%%%%%%%%%%先验概率%%%%%%%%%%%%%%
       P(1)=size(data_1,1)/size(data,1);
       P(2)=size(data_2,1)/size(data,1);
       P(3)=size(data_3,1)/size(data,1);
%%%%%%%%%%计算样本均值%%%%%%%%%%
E1=mean(data_E1(:,1:4),1); %选出来的训练集的1-4列，各求均值
E2=mean(data_E2(:,1:4),1); %1代表返回每一列元素均值的行向量，即列的均值横着放1*4
E3=mean(data_E3(:,1:4),1);
%%%%%%%%%%计算样本协方差矩阵%%%%%%%%%%%%% 
X1=data_E1(:,1:4)-repmat(E1,size(data_E1(:,1:4),1),1);  %X1=x-μ，μ是x属于wi类的期望
X2=data_E2(:,1:4)-repmat(E2,size(data_E2(:,1:4),1),1);  %E2矩阵堆叠在1*1矩阵中
X3=data_E3(:,1:4)-repmat(E3,size(data_E3(:,1:4),1),1);
cov1=1/(size(data_E1(:,1:4),1))*X1'*X1;%（1/n）[（x-miu）*(x-miu)转置]
cov2=1/(size(data_E2(:,1:4),1))*X2'*X2;
cov3=1/(size(data_E3(:,1:4),1))*X3'*X3;

%%%%%%%%%%%%测试%%%%%%%%%%%%%
%%%%%%%%%%%类条件概率%%%%%%%%%%%%
d=4;      %测试样本特征
 for i=1:size(data2,1)
       p1(i)=1/((2*pi)^(d/2)*det(cov1)^(1/2))*exp(-1/2*(data2(i,1:4)-E1)*cov1^(-1)*(data2(i,1:4)-E1)');
end
for i=1:size(data2,1)
       p2(i)=1/((2*pi)^(d/2)*det(cov2)^(1/2))*exp(-1/2*(data2(i,1:4)-E2)*cov2^(-1)*(data2(i,1:4)-E2)');
end
for i=1:size(data2,1)
       p3(i)=1/((2*pi)^(d/2)*det(cov3)^(1/2))*exp(-1/2*(data2(i,1:4)-E3)*cov3^(-1)*(data2(i,1:4)-E3)');
end
%%%%%%%%%%%联合概率密度(后验概率)%%%%%%%%%%%
P1=p1*P(1);    %后验概率=（类条件概率*先验概率）/全概率密度
P2=p2*P(2);
P3=p3*P(3);
%% 分类结果
%%%%%%按多类情况处理%%%%%
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
       T_Y(i)=1;
       elseif T_P(i)==P2(i)
          T_Y(i)=2;
       elseif T_P(i)==P3(i)
          T_Y(i)=3;
       end
 end
 %%%%%%%%%%%%正确计数%%%%%%%%
 count_Ture=0;
for i=1:size(data2,1)
       if T_Y(i)==data2(i,5)
          count_Ture=count_Ture+1;
       end
end
fprintf('分类准确率：%f%%\n',count_Ture/size(data2,1)*100)