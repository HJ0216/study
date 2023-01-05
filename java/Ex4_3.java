public class Ex4_3 {
    public static void main(String[] args){
        int sum2=0;
        for(int i=1; i<=10; i++){
            for(int j=1; j<=i; j++){
                sum2 += j;
            }
        } System.out.println(sum2);
    }
    
}
