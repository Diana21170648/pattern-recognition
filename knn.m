clear all；%读取数据
data1=xlsread('C:\Users\Administratsr\OneDrive\桌面\模式识别程序\XIN.xlsx');
data=data1(randperm(size(data1, 1)),:);
ratio=0.25;%测试数据所占比例
[N,M]=size(data);%624*5
K=2;%K表示所取某一邻近范围内数据的个数，,K=4的时候准确率在76%左右，K=2准确率82%
dataX=data(:,1:4);%624*4
dataY=data(:,5);%624*1
num_test=N*ratio;%测试数据为625*0.25=156个
%归一化处理newData=(oldData-minValue)/(maxValue-minValue);
minValue=min(dataX);
maxValue=max(dataX);
dataX=(dataX-repmat(minValue,N,1))./(repmat(maxValue-minValue,N,1));%把数据归到0-1的区间，提高计算速度，减小计算误差
count_Ture=0;%
for i=1:num_test %1:156
    idx=KNN(dataX(num_test+1:N,:),dataY(num_test+1:N,:),dataX(i,:),K);%众数=KNN
    %fprintf('该测试数据的真实类为：%d\n',dataY(i,:));
    if idx==dataY(i,:) %输出=真实值，则正确个数加1
        count_Ture=count_Ture+1;
    end
end
fprintf('准确率为：%f\n',(count_Ture/num_test)*100);

function [ idx ] = KNN( dataX,dataY,testData,K )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[N,M]=size(dataX);
%计算训练数据集与测试数据之间的欧氏距离dist
dist=zeros(N,1);
for i=1:N %i指测试集数据的个数
    dist(i,:)=norm(dataX(i,:)-testData);%欧氏距离，norm表示求范数
end
%将dist从小到大进行排序
[Y,I]=sort(dist,1);   
K=min(K,length(Y));%计算欧氏距离后，取最近的K个点的最小值
%将训练数据对应的类别(1.2.3)与训练数据排序结果对应
labels=dataY(I);
%{
%确定前K个点所在类别的出现频率
classNum=length(unique(trainClass));%取集合中的单值元素的个数
labels=zeros(1,classNum);
for i=1:K
    j=trainClass(i);
    labels(j)=labels(j)+1;
end
%返回前K个点中出现频率最高的类别作为测试数据的预测分类
[~,idx]=max(labels);
%}
%确定前K个点所在类别的出现频率
idx=mode(labels(1:K));%mode函数求众数
%fprintf('该测试数据属于类 %d  ',idx);
end