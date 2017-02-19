clc;
clear all;

RGB = imread('house.jpg');
%figure;
%imshow(RGB);

LAB = rgb2lab(RGB); %convert RGB to LAB
%figure;
%imshow(LAB);

RGB2 = lab2rgb(LAB); %try to recovery RGB
%figure;
%imshow(RGB2);

figure, 
subplot(131); imshow(RGB, []); 
subplot(132); imshow(LAB, []);
subplot(133); imshow(RGB2, []);