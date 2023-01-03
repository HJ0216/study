public class Ex3_1{
    public static void main(String[] args) {
        int x = 2;
        int y = 6;

        char c = 'A';

        System.out.println(1+x<<33); // (1+2)*2^1
        System.out.println(y>=5 || x<0 && x>2); // true, 연산 순서: && -> ||
        System.out.println(y += 10 - x++); // x++:2, 10-x++:8, y+10-x++:14
        // x++: x를 이용한 첫번째 연산은 x 값, 그 다음 x를 이용한 연산은 x+1
        // ++x: x를 이용한 첫번째 연산부터 x+1
        // System.out.println(x++); // 2
        // System.out.println(x); // 3
        // System.out.println(++y); // 6
        // System.out.println(y); // 6
        System.out.println(x+=2); // 변수를 계속 이용하는 경우는 값이 덮어쓰기 됨(x=3)
        System.out.println(!('A'<c && c<='Z')); // false
        System.out.println('C'-c); // 0
        System.out.println('5'-'0');












        // int j = 1;
        // System.out.println(j<<2); // 1*2^2
        // System.out.println(j<<30); // 1073741824
        // System.out.println(j<<31); // -2147483648
        // System.out.println(j<<32); // 1*2^0
    }
}