function pk2pk = peak2peak(Y)

[r c] = size(Y);

if r>c
    Y = Y';
    r = c;
end


for i = 1:r
    upks{i}   = findpeaks(Y(i,:));
    lpks{i}   = findpeaks(-Y(i,:));
    us        = sort(upks{i});
    us        = us(ceil(length(us)*0.99));
    ls        = sort(lpks{i});
    ls        = ls(ceil(length(ls)*0.99));    
    pk2pk(i)  = ls + us;
end
 
pk2pk = mean(pk2pk);
