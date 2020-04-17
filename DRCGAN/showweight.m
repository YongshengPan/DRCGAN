cls = load('cls90_AB.mat');

figure(1);
 for idx = 61:102%62%
     im = squeeze(cls.iB(:,:,idx)/2+0.5);
     im = cat(3, im, im, im);
    mk = squeeze(cls.mB(:,:,idx))*256*1.25;
    mk = label2rgb(gray2ind(mk), jet(255));
     imshow(im*0.5+im2single(mk)*0.5);
     imwrite(im*0.5+im2single(mk)*0.5, 'B1_90.jpg');
     pause(0.1); 
 end
 
figure(2); 
for idx = 51:82%102%
    im = squeeze(cls.iB(:,idx,:)/2+0.5);
    im = cat(3, im, im, im);
    mk =squeeze(cls.mB(:,idx,:))*256*1.25;
    mk = label2rgb(gray2ind(mk), jet(255));
    imshow(im*0.5+im2single(mk)*0.5);
    imwrite(im*0.5+im2single(mk)*0.5, 'B2_90.jpg');
    pause(0.1);
end

figure(3);
for idx = 51:76%72
    im = squeeze(cls.iB(idx,:,:)/2+0.5);
    im = cat(3, im, im, im);
    mk =squeeze(cls.mB(idx,:,:))*256*1.25;
    mk = label2rgb(gray2ind(mk), jet(255));
    imshow(im*0.5+im2single(mk)*0.5);
    imwrite(im*0.5+im2single(mk)*0.5, 'B3_90.jpg');
    pause(0.1);
end

figure(4);
cb = zeros(255, 5, 3);
cb(:,1,:) = jet(255);
cb(:,2,:) = jet(255);
cb(:,3,:) = jet(255);
cb(:,4,:) = jet(255);
cb(:,5,:) = jet(255);
imshow(flipud(cb));
imwrite(cb, 'cb.jpg');


