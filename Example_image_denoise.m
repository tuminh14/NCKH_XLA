function image = Example_image_denoise(a,file_name)
%     clear all
%     close all
%     clc

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Demo file for deconvtv
    % Image 'salt and pepper' noise removal
    % 
    % Stanley Chan
    % University of California, San Diego
    % 20 Jan, 2011
    %
    % Copyright 2011
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Prepare images
    f_orig  = im2double(imread(char(a)));
   [rows cols frames] = size(f_orig);
    H       = fspecial('gaussian', [9 9], 2);
%     g       = imfilter(f_orig, H, 'circular');
%     g       = imnoise(g, 'salt & pepper', 0.05);

    % Setup parameters (for example)
    opts.rho_r   = 5;
    opts.rho_o   = 100;
    opts.beta    = [1 1 0];
    opts.print   = true;
    opts.alpha   = 0.7;
    opts.method  = 'l1';

    % Setup mu
    mu           = 20;

    % Main routine
    tic
    out = deconvtv(f_orig, H, mu, opts);
    image = uint8(out.f*255);
    path = strcat('Deconvtv_out/',file_name);
    out_path = char(path);
    imwrite(image,out_path);
    disp("deconved");
    toc 
end
% % Display results
% figure(1);
% imshow(g);
% title('input');
% 
% figure(2);
% imshow(out.f);
% title('output');