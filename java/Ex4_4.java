public class Ex4_4 {
    public static void main(String[] args){
        int sum3 = 0;
        int num = 1;
        while(sum3<100){
            if(num%2==0){
                sum3-=num;
            } else {
            sum3 += num;
            } num++;
        } System.out.println("num and sum3: "+num+", "+sum3);
    }
}





