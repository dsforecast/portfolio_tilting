function results=testings(data,type)

if type==1
    cv=[2;1;0.5];
elseif type==2
    cv=[0.8;0.5;0.2];
elseif type==3
    cv=[2;1;0.5];
end

for i=1:size(data,1)
    for j=1:size(data,2)    
        if data(i,j)>=cv(1,1)
            results(i,j)=0.0099;
        elseif data(i,j)<cv(1,1) && data(i,j)>=cv(2,1)
            results(i,j)=0.0499;
        elseif data(i,j)<cv(2,1) && data(i,j)>=cv(3,1)
            results(i,j)=0.099;
        else
            results(i,j)=0.2;
        end
    end
end
                
end

