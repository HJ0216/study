public class Ex4_9 {
    public static void main(String[] args) {
        String str = "12345";
        int num = 0;

        for(int i=0; i<str.length(); i++){
            num += str.charAt(i)-'0';
        } System.out.println(num);

        // System.out.println(str.charAt(1)-'0');
        // str.charAt(1)='2'=50
        // -'0'을 통해서 -48 수행
        
    }
}
